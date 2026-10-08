"""Execute the ORIGINAL admission function in isolation; no GPU mutation."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


class Guard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path('/root/autodl-tmp/src/sglang-main/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py')
        tree=ast.parse(path.read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_can_run_ragged_verify_graph')
        for arg in node.args.args:arg.annotation=None
        node.returns=None
        isolated=ast.Module(body=[node],type_ignores=[])
        env={}
        exec(compile(ast.fix_missing_locations(isolated),str(path),'exec'),env)
        cls.original_guard=staticmethod(env[node.name])

    def layout(self,lens=[8],total=8,tokens=8):
        return NS(bs=len(lens) if lens is not None else 1, verify_lens_cpu=lens,
            graph_num_tokens=tokens,total_verify_tokens=total,
            verify_lens=NS(sum=lambda:NS(item=lambda:8)))

    def runner(self,capture=True):
        return NS(attn_backend=NS(supports_ragged_verify_graph=True),
            capture_num_tokens=[8],_ragged_capture_slots=lambda _:1,
            _captured_ragged_layouts={8:self.layout()} if capture else {},
            require_mlp_sync=False,is_encoder_decoder=False,capture_hidden_mode=1)

    def call(self,layout=None,bs=1,runner=None,hidden=1):
        return self.original_guard(runner or self.runner(),
            NS(batch_size=bs,capture_hidden_mode=hidden),layout or self.layout())

    def test_exact_only(self):self.assertTrue(self.call())
    def test_multirow_same_total(self):self.assertFalse(self.call(self.layout([4,4]),bs=2))
    def test_partial_not_full_tier(self):self.assertFalse(self.call(self.layout([7],7)))
    def test_device_only_not_admitted(self):self.assertFalse(self.call(self.layout(None)))
    def test_missing_capture(self):self.assertFalse(self.call(runner=self.runner(False)))
    def test_tokens_outside_catalog(self):self.assertFalse(self.call(self.layout([9],9,9)))
    def test_hidden_mode(self):self.assertFalse(self.call(hidden=2))
    def test_policy_not_relaxing_guard(self):
        text=Path(__file__).with_name('exact_target_graph_policy.py').read_text()
        self.assertIn('allowed = original_guard(self, fb, layout)',text)
        self.assertIn('return allowed',text)
        self.assertIn('self._decode_graph_allowed = False',text)
        self.assertNotIn('RMSNorm.forward_native =',text)


if __name__ == '__main__':unittest.main()
