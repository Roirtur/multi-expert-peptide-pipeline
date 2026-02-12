# Quick Reference Card: PlaceHolders and Data Milestone

**Due Date:** February 23, 2026 | **Duration:** 11 days | **Issues:** 6

---

## 📋 Issues at a Glance

| # | Issue | Priority | Time | Start | Blocked By |
|---|-------|----------|------|-------|------------|
| 2 | Base Classes | 🔴 High | 2-3d | Day 1 | None |
| 6 | Standards | 🔴 High | 2d | Day 2 | None |
| 1 | Main Loop | 🔴 High | 3-4d | Day 4 | #2, #3 |
| 3 | Logging | 🟡 Med | 2-3d | Day 4 | None |
| 4 | Dataset | 🔴 High | 3-4d | Day 5 | None |
| 5 | Data Refine | 🟡 Med | 2-3d | Day 8 | #4 |

---

## 🎯 Daily Targets

| Day | Target |
|-----|--------|
| 1 | Start Issue 2 (Base Classes) |
| 2 | Start Issue 6 (Standards) |
| 3 | Complete Issues 2 & 6 |
| 4 | Start Issues 1 (Main Loop) & 3 (Logging) |
| 5 | Start Issue 4 (Dataset) |
| 7 | Complete Issues 1, 3 |
| 8 | Complete Issue 4, Start Issue 5 |
| 10 | Complete Issue 5 |
| 11 | Integration testing |

---

## 🔑 Key Deliverables

### Issue 1: Main Loop
- `src/pipeline/orchestrator.py`
- Pipeline config schema
- Integration tests

### Issue 2: Base Classes
- `src/agents/base_agent.py`
- `src/agents/ai_agent/base_ai_agent.py`
- `src/agents/chem_agent/base_chem_agent.py`
- `src/agents/bio_agent/base_bio_agent.py`

### Issue 3: Logging
- `src/utils/logging.py`
- Execution tracer
- Metrics collector

### Issue 4: Dataset
- `data/processed/` with train/val/test
- Data loading utilities
- Dataset documentation

### Issue 5: Refinement
- Data cleaning pipeline
- Feature engineering
- Quality assessment

### Issue 6: Standards
- Data schemas
- JSON validation schemas
- Format converters

---

## ⚠️ Key Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Dataset access | High | Start early, have backups |
| Integration issues | Medium | Clear interfaces, test incrementally |
| Time pressure | Medium | Prioritize ruthlessly |

---

## ✅ Definition of Done

- [ ] All code merged
- [ ] Tests pass (>90% coverage)
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] No blocking issues
- [ ] Integration tests pass

---

## 📞 Quick Contacts

- **Architecture:** Issues 1, 2, 6
- **Infrastructure:** Issue 3
- **Data:** Issues 4, 5

---

## 🔗 Links

- **Milestone:** github.com/Roirtur/multi-expert-peptide-pipeline/milestone/2
- **Detailed Docs:** MILESTONE_PLACEHOLDERS_AND_DATA_ISSUES.md
- **Summary:** ISSUES_SUMMARY.md
- **Visual:** MILESTONE_VISUAL_OVERVIEW.md

---

## 📊 Progress Tracker

```
Issue 2: Base Classes          [ ] Not Started  [ ] In Progress  [ ] Done
Issue 6: Standards             [ ] Not Started  [ ] In Progress  [ ] Done
Issue 1: Main Loop             [ ] Not Started  [ ] In Progress  [ ] Done
Issue 3: Logging               [ ] Not Started  [ ] In Progress  [ ] Done
Issue 4: Dataset               [ ] Not Started  [ ] In Progress  [ ] Done
Issue 5: Data Refinement       [ ] Not Started  [ ] In Progress  [ ] Done
```

---

**Print this card and keep it visible during the sprint!**

---

_Version 1.0 | Created: Feb 12, 2026 | Next Update: Weekly_
