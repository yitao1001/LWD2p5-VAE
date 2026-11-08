"""新参考路线的行为测试；所有伪造权重仅用于接口验证，不能用于科研训练。"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torchvision.models import alexnet

from lwd_vae.dataset import LWDVAEDataset
from lwd_vae.finetuning.config import PROJECT_ROOT, load_config, resolve_paths, validate_config
from lwd_vae.finetuning.features import AlexNetFeatureL2, AlexNetLPIPS
from lwd_vae.finetuning.losses import VAELoss, loss_coefficients
from lwd_vae.finetuning.model import CompVisKLF8, Encoder, Decoder, DiagonalGaussianDistribution, upstream_provenance
from lwd_vae.finetuning.trainer import train
from lwd_vae.inference import LWDVAE
from lwd_vae.runtime import load_checkpoint
from lwd_vae.weights import load_pretrained
from test_vae import run as evaluate

ROOT = PROJECT_ROOT
DATA = ROOT / 'datasets/smoke_global_v2_2'


class FinetuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(prefix='lwd_reference_tests_')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.work = Path(cls.temp.name)
        cls.config = load_config('configs/finetuning/smoke_original_core.json')
        cls.config['training'].update(epochs=1, max_train_samples=2, max_val_samples=2)
        cls.config['model']['dropout'] = 0.1
        cls.alex_path = cls.work / 'synthetic_alexnet.pth'
        torch.manual_seed(10)
        torch.save(alexnet(weights=None).state_dict(), cls.alex_path)
        cls.vae_path = cls.work / 'synthetic_vae.pt'
        torch.save(CompVisKLF8(cls.config['model']).state_dict(), cls.vae_path)

    def make_config(self, name, feature='none'):
        config = deepcopy(self.config)
        config['paths'].update(output=str(self.work / name), dataset=str(DATA))
        if feature != 'none':
            config['loss'].update(feature=feature, feature_weight=1.0)
            config['paths']['alexnet_weights'] = str(self.alex_path)
        return config

    def assert_state_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_state_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_state_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_default_requires_pretraining_and_paths_do_not_depend_on_cwd(self):
        config = load_config()
        self.assertEqual(config['initialization']['mode'], 'pretrained')
        self.assertEqual(config['model']['base_channels'], 128)
        result = subprocess.run([sys.executable, str(ROOT / 'train_vae.py'), '--print-config'],
                                cwd=self.work, capture_output=True, text=True, check=True)
        actual = json.loads(result.stdout)
        self.assertEqual(actual['resolved_paths']['dataset'], str(ROOT / 'datasets/lwd_global_v2_2'))
        bad = deepcopy(config)
        bad['loss']['misspelled_weight'] = 1
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_vendor_source_is_unmodified_except_import_prefix(self):
        base = ROOT / 'lwd_vae/vendor/compvis'
        for record in upstream_provenance()['files']:
            data = (base / record['path']).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), record['local_sha256'])
            upstream = data.replace(b'from lwd_vae.vendor.compvis.ldm.', b'from ldm.')
            self.assertEqual(hashlib.sha256(upstream).hexdigest(), record['upstream_sha256'])

    def test_wrapper_matches_original_autoencoder_methods(self):
        # 执行审计副本中未修改的构造/encode/decode/forward 方法，剥离 Lightning 生命周期。
        tree = ast.parse((ROOT / 'references/compvis/autoencoder_original.py').read_text())
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AutoencoderKL')
        original.bases = [ast.Attribute(ast.Name('nn', ast.Load()), 'Module', ast.Load())]
        original.body = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in ('__init__', 'encode', 'decode', 'forward')]
        module = ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[]))
        namespace = dict(torch=torch, nn=nn, Encoder=Encoder, Decoder=Decoder,
                         DiagonalGaussianDistribution=DiagonalGaussianDistribution,
                         instantiate_from_config=lambda _: nn.Identity())
        exec(compile(module, '<upstream AutoencoderKL methods>', 'exec'), namespace)
        spec = self.config['model']
        original_model = namespace['AutoencoderKL'](dict(double_z=True, z_channels=4,
            resolution=128, in_channels=1, out_ch=1, ch=32, ch_mult=(1, 2, 4, 4),
            num_res_blocks=1, attn_resolutions=[], dropout=spec['dropout']), {}, 4).eval()
        adapter = CompVisKLF8(spec).eval()
        adapter.load_state_dict(original_model.state_dict(), strict=True)
        x = torch.randn(1, 1, 48, 120)
        with torch.no_grad():
            a, pa = adapter(x, sample_posterior=False)
            b, pb = original_model(x, sample_posterior=False)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(pa.mean, pb.mean, rtol=0, atol=0)
        self.assertEqual(tuple(pa.mean.shape), (1, 4, 6, 15))

    def test_rgb_average_and_reject_partial_weights_without_mutation(self):
        model = CompVisKLF8(self.config['model'])
        state = {k: v.clone() for k, v in model.state_dict().items()}
        for key, axis in (('encoder.conv_in.weight', 1), ('decoder.conv_out.weight', 0), ('decoder.conv_out.bias', 0)):
            state[key] = torch.cat([state[key], state[key] * 2, state[key] * 3], dim=axis)
        path = self.work / 'rgb.pt'
        torch.save({'state_dict': state}, path)
        report = load_pretrained(model, path, rgb_to_gray='mean')
        self.assertEqual(report['adapted_tensor_count'], 3)
        torch.testing.assert_close(model.encoder.conv_in.weight, state['encoder.conv_in.weight'].mean(1, keepdim=True))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        del state['quant_conv.weight']
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, 'no weights loaded'):
            load_pretrained(model, path, rgb_to_gray='mean')
        self.assert_state_equal(before, model.state_dict())

    def test_kl_reduction_and_feature_schedule(self):
        settings = deepcopy(self.config['loss'])
        prediction = torch.ones(2, 1, 48, 120, requires_grad=True)
        mean = torch.ones(2, 4, 6, 15, requires_grad=True)
        posterior = DiagonalGaussianDistribution(torch.cat([mean, torch.zeros_like(mean)], dim=1))
        target = torch.zeros_like(prediction)
        summed = VAELoss(settings, None)(prediction, target, posterior, 0)
        settings['kl_reduction'] = 'mean_per_element'
        averaged = VAELoss(settings, None)(prediction, target, posterior, 0)
        self.assertEqual(summed['kl'].item(), 180)
        self.assertEqual(averaged['kl'].item(), 0.5)
        torch.testing.assert_close(summed['loss'], summed['pixel'] + 1e-5 * summed['kl'])
        summed['loss'].backward()
        self.assertGreater(mean.grad.abs().sum().item(), 0)
        inverse = load_config('configs/finetuning/inverse_scattering_reference.json')['loss']
        self.assertEqual(loss_coefficients(inverse, 6), (1.0, 1e-5))
        self.assertEqual(loss_coefficients(inverse, 7), (0.1, 1e-5))

    def test_both_frozen_feature_losses_keep_input_gradient_without_download(self):
        with patch('torch.hub.download_url_to_file', side_effect=AssertionError('implicit download')):
            for kind in ('alexnet_l2', 'lpips_alex'):
                with self.subTest(kind=kind):
                    settings = self.make_config('unused', kind)['loss']
                    criterion = VAELoss(settings, self.alex_path)
                    criterion.train()
                    self.assertFalse(criterion.feature_network.training)
                    prediction = torch.rand(1, 1, 48, 120, requires_grad=True)
                    target = torch.zeros_like(prediction)
                    posterior = DiagonalGaussianDistribution(torch.zeros(1, 8, 6, 15))
                    terms = criterion(prediction, target, posterior, 0)
                    torch.testing.assert_close(terms['weighted_feature'], terms['feature'])
                    terms['feature'].backward()
                    self.assertTrue(torch.isfinite(prediction.grad).all())
                    self.assertGreater(prediction.grad.abs().sum().item(), 0)
                    self.assertTrue(all(p.grad is None and not p.requires_grad for p in criterion.parameters()))
                    # 自比较必须为零；两条路线均不能把特征权重误接到 KL 系数。
                    self.assertAlmostEqual(criterion.feature_network(target, target).item(), 0, places=7)

    def test_lpips_agrees_with_official_trunk_and_calibration(self):
        import lpips
        source = alexnet(weights=None)
        source.load_state_dict(torch.load(self.alex_path, weights_only=True))
        # 官方 LPIPS 自己切分 AlexNet，独立检查适配器的五个 slice 没有错位。
        with patch('lpips.pretrained_networks.tv.alexnet', return_value=source):
            reference = lpips.LPIPS(net='alex', pnet_rand=True, verbose=False).eval()
        adapter = AlexNetLPIPS(self.alex_path)
        x, y = torch.randn(1, 1, 48, 120), torch.randn(1, 1, 48, 120)
        with torch.no_grad():
            expected = reference(x.repeat(1, 3, 1, 1), y.repeat(1, 3, 1, 1)).mean()
            actual = adapter(x, y)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_csem_feature_l2_matches_explicit_imagenet_activations(self):
        adapter = AlexNetFeatureL2(self.alex_path, [1, 4, 7, 9, 11])
        source = alexnet(weights=None)
        source.load_state_dict(torch.load(self.alex_path, weights_only=True))
        x, y = torch.randn(1, 1, 48, 120), torch.randn(1, 1, 48, 120)
        a = ((x + 1) / 2).repeat(1, 3, 1, 1)
        b = ((y + 1) / 2).repeat(1, 3, 1, 1)
        mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
        a, b = (a - mean) / std, (b - mean) / std
        losses = []
        with torch.no_grad():
            for index, layer in enumerate(source.features[:12]):
                a, b = layer(a), layer(b)
                if index in (1, 4, 7, 9, 11):
                    losses.append((a-b).square().mean())
            torch.testing.assert_close(adapter(x, y), torch.stack(losses).mean(), rtol=0, atol=0)

    def test_pretrained_training_resume_with_ema_and_features_is_exact(self):
        config = self.make_config('first', 'lpips_alex')
        config['initialization']['mode'] = 'pretrained'
        config['paths']['vae_weights'] = str(self.vae_path)
        config['loss']['pixel'] = 'l1'
        config['ema']['decay'] = .99
        train(config, device_name='cpu')
        first = self.work / 'first/checkpoints/last.pt'
        uninterrupted = self.work / 'continuous'
        train(config, output=uninterrupted, epochs=2, device_name='cpu')
        resumed = self.work / 'continued'
        # 完整 loss_state 必须足以恢复，不能再依赖外部初始权重。
        unavailable = self.work / 'hidden_alexnet.pth'
        self.alex_path.rename(unavailable)
        try:
            with patch('lwd_vae.finetuning.trainer.load_pretrained', side_effect=AssertionError('reload initial weights')):
                train(resume=first, output=resumed, epochs=2, device_name='cpu')
        finally:
            unavailable.rename(self.alex_path)
        _, a = load_checkpoint(resumed / 'checkpoints/last.pt')
        _, b = load_checkpoint(uninterrupted / 'checkpoints/last.pt')
        for key in ('model_state', 'loss_state', 'optimizer_state', 'ema_state', 'best_val_rmse', 'best_epoch', 'global_step'):
            self.assert_state_equal(a[key], b[key])
        for x, y in zip(a['history'], b['history']):
            self.assertEqual({k:v for k,v in x.items() if k!='seconds'}, {k:v for k,v in y.items() if k!='seconds'})
        self.assertEqual(a['ema_state']['num_updates'].item(), 2)
        self.assertEqual(a['format'], 'lwd_vae_finetuning_v2')

    def test_csem_training_checkpoint_evaluation_and_latent_gradient(self):
        config = self.make_config('csem_integration', 'alexnet_l2')
        train(config, device_name='cpu')
        checkpoint = self.work / 'csem_integration/checkpoints/best.pt'
        wrapper = LWDVAE(checkpoint, device='cpu')
        with LWDVAEDataset(DATA, 'test', as_torch=True) as dataset:
            lo, hi = dataset.log10_bounds
            raw = lo + (dataset[0].unsqueeze(0) + 1) * (hi-lo) / 2
        latent = wrapper.encode(raw).mode().detach().requires_grad_(True)
        decoded = wrapper.decode(latent)
        vector = wrapper.decode_inversion_vector(latent)
        np.testing.assert_array_equal(vector.detach().numpy()[0], decoded.detach().numpy()[0, 0].ravel(order='F'))
        vector.square().mean().backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(latent.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in wrapper.parameters()))
        report = evaluate(argparse.Namespace(dataset=DATA, checkpoint=checkpoint,
            output=self.work / 'evaluation', splits=None, device='cpu', batch_size=2,
            workers=0, max_samples=2, save_reconstructions=True, no_plots=True))
        self.assertTrue(np.isfinite(report['splits']['test']['overall']['rmse_log10']))
        self.assertEqual(set(report['splits']), {'test', 'ood_contrast'})

    def test_missing_or_partial_weights_fail_before_output_creation(self):
        config = self.make_config('must_not_exist')
        config['initialization']['mode'] = 'pretrained'
        config['paths']['vae_weights'] = str(self.work / 'missing.pt')
        with self.assertRaises(FileNotFoundError):
            train(config, device_name='cpu')
        self.assertFalse(resolve_paths(config)['output'].exists())
        torch.save({'quant_conv.bias': torch.zeros(8)}, config['paths']['vae_weights'])
        with self.assertRaisesRegex(ValueError, 'no weights loaded'):
            train(config, device_name='cpu')
        self.assertFalse(resolve_paths(config)['output'].exists())


if __name__ == '__main__':
    unittest.main()
