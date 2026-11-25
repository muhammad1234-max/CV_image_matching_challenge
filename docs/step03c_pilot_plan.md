# Pilot plan

Select intact ETs, the smallest development collection (22 images, at most 231 unordered pairs). The preflight validates only these 22 original image hashes against the prior input inventory. The notebook embeds their identities and never loads scoring labels or organizer caches. Its isolated subprocess has an 1800-second wall limit.

Execution sequence after dependency resolution: verify checkpoint bytes, load actual models, run retrieval and pair selection, published hybrid matching, geometric verification and COLMAP reconstruction, assign scenes from models, serialize original image identities and world-to-camera poses, then run both frozen scores in an isolated evaluator. These downstream stages have no implementation acceptance or measurements yet.

Same-dataset comparator: organizer mAA 0.288462, clusterness 0.431818, combined 0.345874, coverage 22/22, successful stages 115.91 seconds, session 156.08 seconds, peak allocated GPU 1694821888 bytes, peak worker RSS 1812656128 bytes. All ns64 measurements and paired differences are pending. The author's ETs example uses the author's evaluator.
