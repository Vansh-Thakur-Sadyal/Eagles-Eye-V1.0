"""Dataset construction for Sentinel AI.

Builds the dataset folder structure from the master write-up (section 46) and
generates the categories the write-up says you must create yourself, because no
public dataset represents them: following, loitering, counter-flow, abandoned
belongings, restricted-zone entry, face occlusion, appearance variation,
incident reasoning and conversational Q&A.

Honesty about what each generator is, and is not, good for
--------------------------------------------------------
Two different kinds of data come out of here, and they have different standing:

1. **Trajectory / event data (simulated).**  Loitering, following, counter-flow,
   restricted entry and abandoned-object scenarios are generated as scripted
   motion with exact ground truth.  Sentinel's behaviour, relationship and
   object agents consume *trajectories*, not pixels, so this data trains and
   evaluates those agents on exactly the representation they use in production.
   It is legitimate training data for that purpose and gives you labels no
   hand-annotated footage would give you for free.

2. **Rendered video (simulated).**  The same scenarios are rendered to MP4.
   This is genuinely useful for end-to-end pipeline testing, threshold tuning
   and demos.  It is *not* a substitute for real footage when training a
   detector: a model trained only on these renders will not generalise to real
   CCTV.  Use real data (Open Images, MOT17, UCF-Crime) for the detector, and
   use these renders to exercise everything downstream of it.

3. **Occlusion and appearance variation (augmented from your images).**  These
   are derived from real photographs you supply, so they carry real texture and
   lighting.  This is the approach the write-up prescribes in S38: augment an
   authorised reference rather than fabricate identities.

4. **Incident reasoning and Q&A (derived from your own recorded incidents).**
   Generated from the incidents Sentinel actually produced, so the reasoning
   targets are real system output rather than invented scenarios.

Every generator writes a `manifest.json` stating which of these it is, so the
provenance of a trained model is never ambiguous.
"""

__all__ = ["behaviour", "occlusion", "reasoning", "structure", "validate"]
