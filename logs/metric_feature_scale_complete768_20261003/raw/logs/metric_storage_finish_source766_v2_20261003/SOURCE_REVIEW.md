# Storage finish launcher revision review

**Verdict: PASS; no blocking findings.** Same-family, provisional; `source-only/not-runtime`.

PASS for the exact revised transport source. It uses the received remote receipt bytes and preserves the original failed attempt, completed retirement, unchanged scientific finish helper and launch guards. No blocking static defect found.

The earlier review missed the Windows CRLF versus remote LF byte mismatch. Its original output and the failed launcher remain historical evidence. This revision review uses the actual failure and exact received bytes; it does not claim successful launch or runtime acceptance.

## Findings

### P1 - PASS - Actual failure is an exact-byte transport mismatch before model launch

Original launcher remote line7 is the receipt SHA assertion, corresponding to launcher source line44. The supplied failure trace reports line7/AssertionError and exit1. Local old receipt is 134866 bytes with 2490 CRLF endings, SHA c19c08147a7564f3efb4cf352b0eacbadd27683656d62e1d8f35539f56d31b54. Received remote receipt is 132376 bytes, LF only, SHA 63956ff16251c212b37ec1264c58878e4ccab46ffed1721f6f8da499ebba4645. Their decoded JSON objects are equal and remote bytes exactly equal old local bytes with CRLF replaced by LF. The original retirement writer reserialized JSON via Windows write_text; the launcher incorrectly treated those bytes as equal to the Linux remote receipt. The failure precedes launcher.log creation (remote line15) and the only Popen (remote line16); the collected diagnosis independently records absent finish/log/launch receipt.

Evidence: `C:/Users/gb/.codex_tmp/retire_closed_m0_766.py:58`; `C:/Users/gb/.codex_tmp/retire_closed_m0_766.py:65-70`; `C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766.py:19-20`; `C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766.py:44-53`; `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/deploy/stderr.txt:1-3`; `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/deploy/EXIT.json:1`; `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/retirement/TRANSPORT_DIAGNOSIS.json:3-13`.

### P2 - PASS - Revision fixes the diagnosed byte comparison without relaxing it

V2 changes retirement_path to the exact received REMOTE_RETIREMENT_BYTES.json. The remote comparison remains a SHA256 equality against the actual remote receipt; its expected digest is now computed from those received bytes, which match the diagnosed remote SHA. No JSON-only acceptance, ignored mismatch, fallback, tolerance change or generic newline-normalization path is introduced. Reading the receipt with json.loads checks status/retired count without rewriting its bytes.

Evidence: `C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766_v2.py:19-22`; `C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766_v2.py:44`; `C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766_v2.py:61`; `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/retirement/TRANSPORT_DIAGNOSIS.json:4-7`.

### P3 - PASS - Completed retirement is reused; original failed launcher is preserved

Both receipt representations record the same completed retirement: exactly24 deleted candidates, 8444122668 retired bytes, completion 2026-10-03T11:40:00.368695+08:00 and 12859166720 free bytes after. The retirement exit receipt is0. V2 reads this existing completion and does not invoke the retirement helper. It uses separate deploy_v2 and remote metric_storage_finish_source766_v2_20261003 locations, leaving original deploy evidence and staged source intact; the original launcher hash is unchanged. This is a corrected transport attempt after a proven pre-Popen failure, not a repeated model/evaluation run or another retirement.

Evidence: `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/retirement/REMOTE_RETIREMENT_BYTES.json:2`; `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/retirement/REMOTE_RETIREMENT_BYTES.json:2487-2489`; `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/retirement/EXIT.json:1`; `C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766_v2.py:15-28`; `C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766_v2.py:50-63`.

### P4 - PASS - Scientific helper and live launch guards are unchanged

The exact diff only redirects the revision review paths, binds the v2 launcher itself in the existing checked script list, selects the exact received receipt, and uses new local/remote attempt directories. retire_closed_m0_766.py, finish_metric_feature_scale766.py, FINISH_SPEC, STORAGE_FINISH_PLAN and retirement/SPEC hashes match the prior source review. V2 still connects only to port2026, verifies uploaded bytes and original failed parent SHA, requires absent finish directory, >=10GiB free disk and physicalGPU1 memory<500MiB, then starts exactly one original tri_reid finish helper with --gpu1. No source265, training, initialization witness, checkpoint, selection, tolerance, evaluator or report code is modified or replaced. The separately reviewed unchanged helper remains responsible for first strict evaluation and all-six verification before one CPU report.

Evidence: `C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766_v2.py:10-17`; `C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766_v2.py:29-63`; `C:/Users/gb/.codex_tmp/finish_metric_feature_scale766.py:37-69`; `C:/Users/gb/.codex_tmp/finish_metric_feature_scale766.py:81-123`.

### P5 - PASS - Review gate and syntax checks are satisfied at source level

V2 requires this review to be PASS or WARN with an empty blocking_findings array and same-family/provisional attribution. It checks reviewed SHA values for unchanged retirement/finish helpers and v2 itself. Both launcher ASTs and embedded remote templates parse; the remote body retains exactly one Popen. All checks used local standard-library read/hash/JSON/AST only; templates were syntax-checked with inert literal substitutions, not executed.

Evidence: `C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766_v2.py:10-17`; `C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766_v2.py:38-68`.

## Acceptance limits

- Narrow transport revision only; retain the earlier whole-source review and its scientific/runtime limitations.
- This review accepts the exact checked v2 source and input bytes, not a launch, evaluation, strict reload, report result or new retirement.
- The retirement completion and no-launch state are claims supported by collected primary receipts/diagnosis; this reviewer made no independent remote query.
- All live capacity, original failure-state and exact-byte gates must still pass when the revised launcher executes.
- Do not rerun retirement or the unchanged failed original launcher. No execution was performed by this review.

No SSH, retirement, experiment imports, model, scorer or report execution occurred. Only the two requested review outputs were written.

## Exact reviewed diff

```diff
--- deploy_metric_storage_finish766.py
+++ deploy_metric_storage_finish766_v2.py
@@ -7,25 +7,25 @@
 import paramiko
 
 base=Path('C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766')
-review_path=base/'reviewer_source/SOURCE_REVIEW.json'
+review_path=base/'launcher_revision/SOURCE_REVIEW.json'
 review=json.loads(review_path.read_bytes())
 assert review['verdict'] in ('PASS','WARN') and review['blocking_findings']==[]
 assert review['review_independence']=='same-family' and review['acceptance_status']=='provisional'
 bound={Path(name).resolve():digest for name,digest in review['reviewed_input_hashes'].items()}
-for name in ('retire_closed_m0_766.py','finish_metric_feature_scale766.py'):
+for name in ('retire_closed_m0_766.py','finish_metric_feature_scale766.py','deploy_metric_storage_finish766_v2.py'):
     path=Path('C:/Users/gb/.codex_tmp')/name
     assert bound[path.resolve()]==hashlib.sha256(path.read_bytes()).hexdigest()
 spec=json.loads((base/'FINISH_SPEC.json').read_bytes())
-retirement_path=base/'retirement/RETIREMENT.json'
+retirement_path=base/'retirement/REMOTE_RETIREMENT_BYTES.json'
 retirement=json.loads(retirement_path.read_bytes())
 assert retirement['status']=='RETIRED_EXACT_24_CLOSED_M0' and retirement['retired_bytes']==8444122668
 assert spec['port']==2026 and spec['root']=='/data/gaob/Re-ID/Trifusion'
-out=base/'deploy'
+out=base/'deploy_v2'
 assert not out.exists();out.mkdir()
 files={'finish_metric_feature_scale766.py':Path('C:/Users/gb/.codex_tmp/finish_metric_feature_scale766.py'),
        'FINISH_SPEC.json':base/'FINISH_SPEC.json','STORAGE_FINISH_PLAN.md':base/'STORAGE_FINISH_PLAN.md',
-       'SOURCE_REVIEW.json':review_path,'SOURCE_REVIEW.md':base/'reviewer_source/SOURCE_REVIEW.md'}
-root=spec['root'];asset=root+'/logs/metric_storage_finish_source766_20261003'
+       'SOURCE_REVIEW.json':review_path,'SOURCE_REVIEW.md':base/'launcher_revision/SOURCE_REVIEW.md'}
+root=spec['root'];asset=root+'/logs/metric_storage_finish_source766_v2_20261003'
 client=paramiko.SSHClient();client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
 client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
 sftp=client.open_sftp();sftp.mkdir(asset)
```

## Exact checked input hashes

| Input | SHA256 |
|---|---|
| `C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766.py` | `1d98146b186fb9a01d332ef260b59986b5f71d66b47281299c12e88538608b42` |
| `C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766_v2.py` | `30ec16057719e5f86970a9e30d82568684c68142f996579529a8f0407dde1f68` |
| `C:/Users/gb/.codex_tmp/finish_metric_feature_scale766.py` | `22807e06f47b387e77218f2dc7ec72393a265d6fd184495542345507749ca2f6` |
| `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/FINISH_SPEC.json` | `8a3bf5d8a9575150f415360fd30dba87da565eeb8531521636432cd71e7e1dfc` |
| `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/STORAGE_FINISH_PLAN.md` | `0408cbd11f3e850b3941816b0687c7d5ebb5180aa2d78d229d345bb2cf384752` |
| `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/deploy/EXIT.json` | `054a76e7744b2c305a4c5b5baccbd7459725cc0b83cba02929a2f737d4fdd757` |
| `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/deploy/stderr.txt` | `2c4cf6ee3fb84976c844e0c6002b84fb287a30845e628c228ba12ddf533560d9` |
| `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/deploy/stdout.json` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/retirement/EXIT.json` | `302c5fcf0227d427ee240abdbd2a948557ea7ab09699f9c3bdeb0f98b67375ce` |
| `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/retirement/REMOTE_RETIREMENT_BYTES.json` | `63956ff16251c212b37ec1264c58878e4ccab46ffed1721f6f8da499ebba4645` |
| `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/retirement/RETIREMENT.json` | `c19c08147a7564f3efb4cf352b0eacbadd27683656d62e1d8f35539f56d31b54` |
| `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/retirement/SPEC.json` | `1a3548c272a353b80e99e5b6e9666fa8c80b3ba5dafcd4d2603ac3749e605642` |
| `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/retirement/TRANSPORT_DIAGNOSIS.json` | `85654c81838275932b27d93616523f9a2174ee8040e098cc3f97c3f1eb6d93a2` |
| `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/retirement/stderr.txt` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/reviewer_source/SOURCE_REVIEW.json` | `08498a277db0d9a13d1e3e3fe866b49ef3fd8f1316efbcec991953ab5fc4f2fa` |
| `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/reviewer_source/SOURCE_REVIEW.md` | `30418c863022859a627c64b14a2d0d99a33e4568df814e8b40ab796cdbd605c2` |
| `C:/Users/gb/.codex_tmp/retire_closed_m0_766.py` | `a65fb7eab10b30162c775ae2d2c596a06179be545c253fd70346c00353ce1979` |
