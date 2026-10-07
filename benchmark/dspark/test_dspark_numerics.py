"""CPU-only guards for the default-OFF precision correctness probe."""
import unittest
from types import SimpleNamespace

from dspark_numerics import configure_bf16_full_reduction


class TestNumerics(unittest.TestCase):
    def setUp(self):
        self.flags = SimpleNamespace(allow_bf16_reduced_precision_reduction=True)
        self.torch = SimpleNamespace(__version__="2.13.0+cu130", bfloat16="bf16",
            cuda=SimpleNamespace(get_device_capability=lambda _: (12, 0)),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=self.flags)))
        self.args = dict(enabled=True, is_dspark=True, device="cuda", dtype="bf16",
            quantization=None, tp_size=1, graphs_disabled=True, deterministic=False,
            backend="torch", torch_module=self.torch)

    def test_default_off_has_no_side_effects(self):
        self.assertFalse(configure_bf16_full_reduction(**{**self.args, "enabled": False, "torch_module": None}))
        self.assertTrue(self.flags.allow_bf16_reduced_precision_reduction)

    def test_on_sets_only_existing_precision_flag(self):
        self.assertTrue(configure_bf16_full_reduction(**self.args))
        self.assertFalse(self.flags.allow_bf16_reduced_precision_reduction)
        self.assertTrue(configure_bf16_full_reduction(**self.args))

    def test_incompatible_geometry_rejected_before_flag_change(self):
        for key, value in (("is_dspark", False), ("device", "cpu"), ("dtype", "fp16"),
                           ("quantization", "fp4"), ("tp_size", 2), ("graphs_disabled", False),
                           ("deterministic", True), ("backend", "cutedsl")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                configure_bf16_full_reduction(**{**self.args, key: value})
            self.assertTrue(self.flags.allow_bf16_reduced_precision_reduction)
        self.torch.__version__ = "other"
        with self.assertRaises(ValueError):
            configure_bf16_full_reduction(**self.args)
        self.torch.__version__ = "2.13.0+cu130"
        self.torch.cuda.get_device_capability = lambda _: (10, 0)
        with self.assertRaises(ValueError):
            configure_bf16_full_reduction(**self.args)
        self.assertTrue(self.flags.allow_bf16_reduced_precision_reduction)


if __name__ == "__main__":
    unittest.main()
