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

    def test_capture_slots_do_not_reject_eager_fallback(self):
        path=Path(__file__).with_name('exact_target_graph_policy.py')
        tree=ast.parse(path.read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='capture_slots')
        env={}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),env)
        runner=NS(ragged_verify_mode=True,model_runner=NS(is_draft_worker=False))
        for tokens in [1,7,8,9,32,64]:self.assertEqual(env['capture_slots'](runner,tokens),1)

    def test_actual_capture_required_and_early_bootstrap(self):
        folder=Path(__file__).parent
        launcher=(folder/'exact_target_launcher.py').read_text()
        self.assertLess(launcher.index('make_hook(configs[0])'),launcher.index('result = original(model_runner'))
        self.assertIn('layout.verify_lens_cpu != [8]',launcher)
        self.assertIn("supports_ragged_verify_graph', False)",launcher)
        self.assertNotIn('supports_ragged_verify_graph =',launcher)
        diagnostic=(folder/'target_graph_diagnostic.py').read_text()
        self.assertIn("env['SGLANG_DSPARK_RAGGED_VERIFY_GRAPH'] = '1'",diagnostic)
        self.assertIn("if not final['actual_execute']",diagnostic)

    def test_host_mirror_uses_actual_not_bucket_values(self):
        from exact_target_graph_policy import exact_host_mirror
        from copy import copy
        def replace(value,**kw):
            out=copy(value)
            for key,item in kw.items():setattr(out,key,item)
            return out
        for length in [1,7,8]:
            device=NS(cpu=lambda:NS(tolist=lambda:[length]))
            layout=NS(bs=1,graph_num_tokens=8,verify_lens_cpu=None,total_verify_tokens=None,
                verify_lens=device,qo_indptr_device=object())
            out,read=exact_host_mirror(layout,1,replace)
            self.assertTrue(read)
            self.assertEqual(out.verify_lens_cpu,[length])
            self.assertEqual(out.total_verify_tokens,length)
            self.assertIs(out.verify_lens,device)
            self.assertIs(out.qo_indptr_device,layout.qo_indptr_device)
            self.assertIsNone(layout.verify_lens_cpu)
            self.assertEqual(self.call(self.layout([length],length)),length==8)
        for bad in [0,9]:
            layout=NS(bs=1,graph_num_tokens=8,verify_lens_cpu=None,
                      verify_lens=NS(cpu=lambda:NS(tolist=lambda:[bad])))
            with self.assertRaises(ValueError):exact_host_mirror(layout,1,replace)

    def test_mirror_multirow_and_known_layout_not_read(self):
        from exact_target_graph_policy import exact_host_mirror
        for bs,tokens,lens in [(4,8,None),(1,16,None),(1,8,[7])]:
            layout=NS(bs=bs,graph_num_tokens=tokens,verify_lens_cpu=lens,verify_lens=None)
            result,read=exact_host_mirror(layout,bs,None)
            self.assertIs(result,layout)
            self.assertFalse(read)


if __name__ == '__main__':unittest.main()
