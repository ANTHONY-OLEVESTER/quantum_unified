# Zenodo Upload Checklist

## Pre-Upload Preparation ✅

- [x] Archive created: `Quantum_Formula_Zenodo_Archive.zip` (53 MB)
- [x] Documentation prepared: `ZENODO_README.md`
- [x] Metadata prepared: `ZENODO_METADATA.json`
- [x] Excludes .venv and quantum_Git_Repo as requested
- [x] All simulation data included
- [x] All figures included
- [x] Paper PDF included
- [x] Source code included

---

## Zenodo Upload Steps

### Step 1: Create Account
- [ ] Go to https://zenodo.org
- [ ] Click "Sign up" (or "Log in" if you have account)
- [ ] **Recommended:** Sign in with ORCID for better academic attribution
- [ ] Alternative: Sign in with GitHub

### Step 2: Start New Upload
- [ ] Click green **"+ New upload"** button (top right)
- [ ] You'll see the upload form

### Step 3: Upload Files
- [ ] Drag and drop `Quantum_Formula_Zenodo_Archive.zip`
- [ ] **OR** click "Choose files" and select the zip
- [ ] Wait for upload to complete (53 MB - should take 1-5 minutes)
- [ ] Zenodo will show the file with a checkmark when done

### Step 4: Fill Basic Information

#### Required Fields:

**Upload type:**
- [ ] Select: **"Dataset"**

**Title:**
- [ ] Enter: `Universal Curvature-Information Principle: Simulation Data and Code`

**Authors:**
- [ ] Click "+ Add creator"
- [ ] Name: `Olevester, Anthony`
- [ ] Affiliation: `Independent Researcher` (or your institution)
- [ ] ORCID: [your ORCID if you have one]

**Description:**
- [ ] Copy from `ZENODO_METADATA.json` or use:
```
Complete simulation data, code, and analysis for the research paper "A Universal
Curvature-Information Principle: Flatness and D^(-1) Concentration under 2-Designs".

This archive contains:
- Python code for computing quantum information-geometric invariants
- Data from 10 phases of numerical simulations
- Publication-quality figures
- LaTeX source for the manuscript

The work demonstrates universal scaling laws (Var(Y) ∝ D^(-1)) and flatness (α → 0)
for the curvature-information invariant Y = sqrt(d_eff - 1) * A^2 / I across chaotic,
structured, and twirled quantum dynamics.
```

### Step 5: License
- [ ] Select: **"Creative Commons Attribution 4.0 International"** (CC-BY-4.0)
- [ ] This is the most open license - allows reuse with attribution
- [ ] Gets more citations than restrictive licenses

### Step 6: Keywords (Add These)
- [ ] quantum information
- [ ] quantum geometry
- [ ] Bures metric
- [ ] Uhlmann fidelity
- [ ] mutual information
- [ ] effective dimension
- [ ] unitary designs
- [ ] quantum thermalization
- [ ] concentration of measure
- [ ] scaling laws
- [ ] entanglement dynamics
- [ ] information geometry
- [ ] quantum chaos
- [ ] 2-design
- [ ] Haar measure
- [ ] quantum simulation

### Step 7: Additional Information (Recommended)

**Version:**
- [ ] Enter: `1.0`

**Language:**
- [ ] Select: `English`

**Related/alternate identifiers:**
- [ ] After your arXiv submission, come back and add:
  - Relation: "is supplement to"
  - Identifier: `arXiv:XXXX.XXXXX`

**Subjects:**
- [ ] Add: `Quantum Physics`
- [ ] Add: `Information Theory`

**References:** (Optional but recommended)
- [ ] Uhlmann, A. (1976). The 'transition probability' in the state space of a ∗-algebra
- [ ] Hübner, M. (1992). Explicit computation of the Bures distance
- [ ] Page, D. N. (1993). Average entropy of a subsystem
- [ ] Dankert, C., et al. (2009). Exact and approximate unitary 2-designs
- [ ] Brandão, F. G., et al. (2016). Local random quantum circuits

**Notes:**
- [ ] Add: `This archive contains all simulation code and data for reproducing the
        results in the paper. Excludes Python virtual environment (.venv) and
        auxiliary git repositories. All simulations can be reproduced using
        Python 3.x with dependencies listed in requirement.txt.`

**Method:**
- [ ] Add: `Numerical simulations using Python with NumPy, SciPy, and custom quantum
        state evolution libraries. Simulations sweep system dimensions D from 2 to
        2^12, test multiple coupling types, and compute information-geometric
        invariants over 10 phases of analysis.`

### Step 8: Communities (Optional)
- [ ] Search and add: "Quantum Physics" (if available)
- [ ] Search and add: "Computational Physics" (if available)

### Step 9: Funding (Optional)
- [ ] Add funding sources if applicable
- [ ] Skip if self-funded/independent research

### Step 10: Review and Publish

**Before clicking Publish:**
- [ ] Double-check title spelling
- [ ] Verify your name is spelled correctly
- [ ] Check that file uploaded successfully
- [ ] Review description for typos

**Understand that:**
- [ ] ✅ Once published, you'll get a **permanent DOI**
- [ ] ✅ Data will be **permanently archived**
- [ ] ✅ You can upload **new versions** later (v2.0, etc.)
- [ ] ⚠️ You **cannot delete** once published (only hide)

**Ready to publish:**
- [ ] Click **"Publish"** button
- [ ] Confirm in the dialog

### Step 11: After Publishing

**Immediately:**
- [ ] Copy the **DOI** (looks like: `10.5281/zenodo.XXXXXXX`)
- [ ] Copy the **citation** text
- [ ] Take screenshot of the published record

**Add DOI to your paper:**
- [ ] Open your LaTeX paper
- [ ] Add in acknowledgments or data availability section:
```latex
\section*{Data Availability}
All simulation data and code are openly available at:
\url{https://doi.org/10.5281/zenodo.XXXXXXX}
```

**Update README files:**
- [ ] Add DOI badge to README.md
- [ ] Update citation section with actual DOI

**Share your work:**
- [ ] Tweet/post about it (optional)
- [ ] Add to your CV/resume
- [ ] Share with collaborators

### Step 12: arXiv Submission
- [ ] Submit paper to arXiv (quant-ph category)
- [ ] Include Zenodo DOI in paper
- [ ] After getting arXiv ID, return to Zenodo:
  - [ ] Click "Edit" on your deposit
  - [ ] Add arXiv identifier as "is supplemented by this upload"
  - [ ] Save changes

---

## Quick Reference URLs

- **Zenodo:** https://zenodo.org
- **Your upload:** https://zenodo.org/uploads (after login)
- **ORCID:** https://orcid.org
- **arXiv:** https://arxiv.org/submit
- **Creative Commons:** https://creativecommons.org/licenses/by/4.0/

---

## Troubleshooting

**Upload fails:**
- Check internet connection
- Try smaller chunks (but your 53MB should be fine)
- Try different browser (Chrome recommended)

**Can't find your upload:**
- Click "Upload" menu → "My uploads"
- Check "My dashboard"

**Need to fix metadata after publishing:**
- Click "Edit" on published record
- Make changes
- Click "Publish" again (creates new version)

**Questions:**
- Check Zenodo FAQ: https://help.zenodo.org
- Contact: support@zenodo.org

---

## Expected Timeline

1. **Upload file:** 1-5 minutes (53 MB)
2. **Fill metadata:** 10-20 minutes (first time)
3. **Review:** 5 minutes
4. **Publish:** Instant
5. **DOI assignment:** Immediate
6. **Indexing (Google Scholar, etc.):** 1-7 days

---

## After Zenodo: Next Steps

- [ ] Submit to arXiv
- [ ] Submit to journal
- [ ] Update CV with publication
- [ ] Share on academic social media
- [ ] Add to Google Scholar profile
- [ ] Update ResearchGate profile (if you have one)

---

## Benefits of Zenodo

✅ **Permanent DOI** - citable identifier
✅ **Free** - no charges for upload or download
✅ **Versioning** - can upload updates as v2.0, v3.0, etc.
✅ **Integration** - works with ORCID, GitHub, arXiv
✅ **Discoverable** - indexed by Google Scholar, DataCite
✅ **Trusted** - hosted by CERN, backed by EU
✅ **Large files** - supports up to 50 GB per dataset
✅ **Open access** - increases citation rates
✅ **Professional** - used by major research projects

---

## Your Files Ready for Upload

- ✅ `Quantum_Formula_Zenodo_Archive.zip` (53 MB) - main archive
- ✅ `ZENODO_README.md` - documentation (inside archive)
- ✅ `ZENODO_METADATA.json` - metadata reference (for your use)
- ✅ `COMPILATION_SUMMARY.md` - project summary
- ✅ `ZENODO_UPLOAD_CHECKLIST.md` - this checklist

---

**Good luck with your upload! 🚀**

Your research represents significant work in quantum information theory,
and making it openly available will benefit the scientific community.

---

*Compiled: November 1, 2025*
