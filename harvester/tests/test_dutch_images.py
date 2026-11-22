import sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dutch_images as d


def marker(name="Testosterone", value=46.8, unit="ng/mg", lo=25, hi=115, status="Within range", comparator=""):
    return {"name": name, "value": value, "comparator": comparator, "unit": unit, "ref_low": lo,
            "ref_high": hi, "status": status, "category": "Androgens"}


class ParsePage(unittest.TestCase):
    def test_dutch_page_markers_get_extractor_ids(self):
        page, problems = d.parse_page({"label": "dutch_page", "report_month": "2023-04",
                                       "collection_month": "2023-03", "age_years": 31, "sex": "male",
                                       "markers": [marker("Estradiol(E2)", 0.94)]})
        self.assertEqual(problems, [])
        self.assertEqual(page["markers"][0]["id"], "estradiol_e2")
        self.assertEqual((page["report_month"], page["age_years"], page["sex"]), ("2023-04", 31, "male"))

    def test_non_dutch_page_keeps_no_markers(self):
        page, _ = d.parse_page({"label": "other_lab", "markers": [marker()]})
        self.assertEqual((page["label"], page["markers"]), ("other_lab", []))

    def test_bad_fields_are_dropped_and_reported(self):
        page, problems = d.parse_page({"label": "dutch_page", "report_month": "April 2023", "age_years": 300,
                                       "sex": "unknown", "markers": [marker(value="high"), "junk", marker()]})
        self.assertEqual(len(page["markers"]), 1)
        self.assertIsNone(page["report_month"]); self.assertIsNone(page["age_years"]); self.assertIsNone(page["sex"])
        self.assertEqual(len(problems), 4)

    def test_ratio_rows_are_kept(self):
        page, problems = d.parse_page({"label": "dutch_page", "markers": [
            marker("2-OH / 16-OH-E1 Balance", 12.87, unit="ratio", lo=2.85, hi=9.88, status="Above range"), marker()]})
        self.assertEqual([m["id"] for m in page["markers"]], ["2_oh_16_oh_e1_balance", "testosterone"])
        self.assertEqual(problems, [])

    def test_unknown_label_is_not_lab(self):
        page, problems = d.parse_page({"label": "selfie"})
        self.assertEqual(page["label"], "not_lab"); self.assertTrue(problems)


KNOWN = {"testosterone", "estradiol_e2", "24hr_free_cortisol", "cortisol_a_waking", "cortisol_b_morning",
         "cortisol_c_afternoon", "cortisol_d_night"}


def page_marker(**kw):
    page, _ = d.parse_page({"label": "dutch_page", "markers": [marker(**kw)]})
    return page["markers"][0]


class Checks(unittest.TestCase):
    def test_clean_markers_pass(self):
        self.assertEqual(d.check_markers([page_marker()], KNOWN), [])

    def test_status_must_agree_with_value_and_range(self):
        self.assertEqual(len(d.check_markers([page_marker(value=200, status="Within range")], KNOWN)), 1)
        self.assertEqual(len(d.check_markers([page_marker(value=10, status="Above range")], KNOWN)), 1)
        self.assertEqual(d.check_markers([page_marker(value=10, status="Below range")], KNOWN), [])
        # the lab compares unrounded values, so a printed value may sit on the limit
        self.assertEqual(d.check_markers([page_marker(value=115, status="Above range")], KNOWN), [])

    def test_a_value_given_as_a_limit_skips_the_status_rule(self):
        self.assertEqual(d.check_markers([page_marker(value=200, comparator="<")], KNOWN), [])

    def test_unknown_names_and_odd_units_are_not_failures(self):
        self.assertEqual(d.check_markers([page_marker(name="Cortisol (U1) - Waking", unit="ng/mg CR")], KNOWN), [])

    def test_l_and_h_flags_mean_below_and_above(self):
        self.assertEqual(d.check_markers([page_marker(value=10, status="L"), page_marker(value=200, status="H")], KNOWN), [])
        self.assertEqual(len(d.check_markers([page_marker(value=10, status="H")], KNOWN)), 1)

    def test_daily_cortisol_must_add_up_to_the_printed_total(self):
        parts = [page_marker(name=n, value=v, lo=None, hi=None, status=None) for n, v in
                 [("Cortisol A (Waking)", 30), ("Cortisol B (Morning)", 100), ("Cortisol C (Afternoon)", 40),
                  ("Cortisol D (Night)", 10)]]
        total = lambda v: page_marker(name="24hr Free Cortisol", value=v, lo=None, hi=None, status=None)
        self.assertEqual(d.check_markers(parts + [total(180)], KNOWN), [])
        self.assertEqual(len(d.check_markers(parts + [total(280)], KNOWN)), 1)

    def test_known_markers_come_from_the_samples(self):
        known = d.known_markers(d.SAMPLES)
        self.assertEqual((known["5a_dht"], known["creatinine_a_waking"]), ("ng/mg", "mg/ml"))


JAN_2024 = 1704100000   # 2024-01-01 UTC
JUN_2024 = 1717200000   # 2024-06-01 UTC


def page(*markers, **head):
    p, _ = d.parse_page({"label": "dutch_page", "markers": list(markers) or [marker()], **head})
    return p


def item(id, author="a1", when=JAN_2024, images=("img1",)):
    return {"id": id, "author_key": author, "created_utc": when, "text": "my DUTCH", "images": list(images)}


def case(author="a1", condition="PSSD", drugs=("sertraline",), when=JAN_2024 - 10, first_person=True):
    return {"author_key": author, "created_utc": when, "is_first_person_case": first_person,
            "condition": condition, "causal_drugs": [{"name": n} for n in drugs], "sex": "male"}


NO = {"on_hormones_at_test": "no", "quote": None}


def assemble(items, pages, cases=(case(),), hormones=None):
    if hormones is None:
        hormones = {i["id"]: NO for i in items}
    return d.assemble(items, pages, hormones, list(cases),
                      {"testosterone": "ng/mg", "estradiol_e2": "ng/mg", "creatinine_a_waking": "mg/ml"})


class Assemble(unittest.TestCase):
    def test_one_post_makes_one_report_from_its_dutch_pages(self):
        pages = {"p1": page(marker(), report_month="2023-12"), "p2": page(marker("Estradiol(E2)", 1.0, lo=0.5, hi=2.2)),
                 "selfie": d.parse_page({"label": "not_lab"})[0]}
        [r] = assemble([item("t3_a", images=["p1", "p2", "selfie"])], pages)
        self.assertEqual(sorted(m["id"] for m in r["markers"]), ["estradiol_e2", "testosterone"])
        self.assertEqual((r["report_month"], r["group"], r["tier"], r["drug"], r["sex"]),
                         ("2023-12", "PSSD", "C", "sertraline", "male"))
        self.assertEqual(r["failures"], [])

    def test_a_post_without_dutch_pages_makes_no_report(self):
        self.assertEqual(assemble([item("t3_a", images=["x"])], {"x": d.parse_page({"label": "other_lab"})[0]}), [])

    def test_identical_repost_by_the_same_author_is_dropped(self):
        pages = {"p1": page(report_month="2023-12"), "p2": page(report_month="2023-12")}
        reports = assemble([item("t3_a", images=["p1"]), item("t1_b", when=JUN_2024, images=["p2"])], pages)
        self.assertEqual([r["item_id"] for r in reports], ["t3_a"])

    def test_same_numbers_from_another_author_are_kept(self):
        pages = {"p1": page(), "p2": page()}
        reports = assemble([item("t3_a", images=["p1"]), item("t3_b", author="a2", images=["p2"])], pages,
                           cases=[case(), case(author="a2")])
        self.assertEqual(len(reports), 2)

    def test_author_needs_a_first_person_pssd_pfs_or_pas_case(self):
        pages = {"p1": page(), "p2": page(), "p3": page()}
        reports = assemble([item("t3_a", images=["p1"]), item("t3_b", author="a2", images=["p2"]),
                            item("t3_c", author="a3", images=["p3"])], pages,
                           cases=[case(condition="none"), case(author="a2", first_person=False),
                                  case(author="a3", condition="PSA", drugs=())])
        self.assertEqual([(r["item_id"], r["group"], r["drug"]) for r in reports], [("t3_c", "PAS", None)])

    def test_group_and_drug_come_from_the_earliest_case(self):
        cases = [case(condition="PFS", drugs=("finasteride",), when=JAN_2024 - 5), case(when=JAN_2024 - 99, drugs=())]
        [r] = assemble([item("t3_a")], {"img1": page()}, cases=cases)
        self.assertEqual((r["group"], r["drug"]), ("PSSD", None))

    def test_missing_report_month_falls_back_to_post_month_with_a_warning(self):
        [r] = assemble([item("t3_a", when=JUN_2024)], {"img1": page()})
        self.assertEqual(r["report_month"], "2024-06")
        self.assertTrue(any("post month" in w for w in r["warnings"]))
        self.assertEqual(r["failures"], [])

    def test_two_pages_disagreeing_on_a_marker_fail_the_report(self):
        pages = {"p1": page(marker(value=40)), "p2": page(marker(value=41))}
        [r] = assemble([item("t3_a", images=["p1", "p2"])], pages)
        self.assertTrue(r["failures"])

    def test_check_failures_and_parse_problems_are_attached(self):
        [r] = assemble([item("t3_a")], {"img1": page(marker(value=500))})
        self.assertTrue(any("marked" in f for f in r["failures"]))

    def test_rows_the_pdf_extractor_does_not_read_are_kept_and_flagged(self):
        pages = {"img1": page(marker(), marker("Cortisol (U1) - Waking", 30, lo=None, hi=None),
                              marker("Cortisol Clearance Rate (CCR)", 2.0, unit="", lo=None, hi=None))}
        [r] = assemble([item("t3_a")], pages)
        got = [(m["id"], m["extractor"], m["loadable"]) for m in r["markers"]]
        self.assertEqual(got, [("testosterone", True, True), ("cortisol_u1_waking", False, True),
                               ("cortisol_clearance_rate_ccr", False, False)])
        self.assertEqual(r["failures"], [])

    def test_a_known_marker_in_another_unit_gets_its_own_id_and_case_is_normalised(self):
        pages = {"img1": page(marker(), marker("Estradiol(E2)", 1.1, unit="ug/gCR", lo=0.5, hi=2.2),
                              marker("Creatinine A (Waking)", 0.9, unit="mg/mL", lo=0.3, hi=3))}
        [r] = assemble([item("t3_a")], pages)
        self.assertEqual([(m["id"], m["unit"]) for m in r["markers"]],
                         [("testosterone", "ng/mg"), ("estradiol_e2_ug_gcr", "ug/gCR"), ("creatinine_a_waking", "mg/ml")])

    def test_a_new_marker_in_two_units_keeps_the_common_unit_and_the_other_gets_its_own_id(self):
        dhea = lambda unit: marker("DHEA", 5, unit=unit, lo=None, hi=None)
        pages = {"p1": page(dhea("ng/mg")), "p2": page(dhea("ng/mg")), "p3": page(dhea("ng/mL"))}
        reports = assemble([item("t3_a", images=["p1"]), item("t3_b", author="a2", images=["p2"]),
                            item("t3_c", author="a3", images=["p3"])], pages,
                           cases=[case(), case(author="a2"), case(author="a3")])
        self.assertEqual([r["markers"][0]["id"] for r in reports], ["dhea", "dhea", "dhea_ng_ml"])

    def test_a_post_with_only_new_markers_still_makes_a_report(self):
        [r] = assemble([item("t3_a")], {"img1": page(marker("Indican", 50, lo=0, hi=80))})
        self.assertEqual(r["markers"][0]["id"], "indican")

    def test_a_report_without_a_hormone_answer_fails(self):
        [r] = assemble([item("t3_a")], {"img1": page()}, hormones={})
        self.assertEqual(r["on_hormones_at_test"], "unknown")
        self.assertTrue(any("hormone" in f for f in r["failures"]))


def report(id, author="a1", month="2024-01", hormones="no", failures=(), when=JAN_2024, **kw):
    return {"item_id": id, "author_key": author, "created_utc": when, "images": ["deadbeef" * 8],
            "group": "PSSD", "tier": "C", "drug": "sertraline", "sex": "male", "age_years": 30,
            "collection_month": None, "report_month": month, "on_hormones_at_test": hormones,
            "hormone_quote": "I was on 100mg test cyp" if hormones == "yes" else None,
            "markers": [dict(page_marker(**kw), extractor=True, loadable=True)], "failures": list(failures),
            "warnings": []}


class Select(unittest.TestCase):
    def ids(self, buckets):
        return {k: [r["item_id"] for r in v] for k, v in buckets.items()}

    def test_default_keeps_the_earliest_report_per_author_and_retests_go_elsewhere(self):
        b = d.select([report("late", month="2024-05"), report("early", month="2023-02"), report("other", author="a2")])
        self.assertEqual(self.ids(b), {"default": ["early", "other"], "retests": ["late"], "hormones": [], "failing": []})

    def test_hormone_users_are_kept_out_of_the_default_export(self):
        b = d.select([report("on_hrt", month="2023-01", hormones="yes"), report("off", month="2024-01"),
                      report("unsure", author="a2", hormones="unknown")])
        self.assertEqual(self.ids(b)["hormones"], ["on_hrt"])
        self.assertEqual(self.ids(b)["default"], ["off", "unsure"])

    def test_failing_reports_are_kept_out_of_the_default_export(self):
        b = d.select([report("bad", month="2023-01", failures=["x"]), report("good", month="2024-01")])
        self.assertEqual((self.ids(b)["failing"], self.ids(b)["default"]), (["bad"], ["good"]))


class Export(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.reports = [report("t3_secretpostid", author="5ecre7auth0rkey"),
                        report("t1_secretcommentid", author="5ecre7auth0rkey", month="2025-01"),
                        report("t3_hrtpostid", author="0therauth0rkey1", hormones="yes")]
        self.reports[0]["images"] = ["https://i.redd.it/secretimage.png"]
        ratio = dict(page_marker(name="2-OH / 16-OH-E1 Balance", value=12.87, unit="ratio", lo=2.85, hi=9.88,
                                 status="Above range"), extractor=False, loadable=False)
        self.reports[0]["markers"].append(ratio)
        d.write_export(d.select(self.reports), self.dir)

    def test_files_per_bucket_with_extractor_columns_and_metadata(self):
        import csv
        names = sorted(p.name for p in self.dir.iterdir())
        self.assertEqual(names, ["dutch_default.csv", "dutch_default_metadata.csv", "dutch_default_other_units.csv",
                                 "dutch_hormones.csv",
                                 "dutch_hormones_metadata.csv", "dutch_retests.csv", "dutch_retests_metadata.csv"])
        rows = list(csv.DictReader(open(self.dir / "dutch_default.csv")))
        self.assertEqual(tuple(rows[0]), d.COLUMNS)
        self.assertEqual((rows[0]["record_id"], rows[0]["marker_id"], rows[0]["value"]), ("rd001", "testosterone", "46.8"))
        other = list(csv.DictReader(open(self.dir / "dutch_default_other_units.csv")))
        self.assertEqual([(r["record_id"], r["marker_id"], r["unit"]) for r in other],
                         [("rd001", "2_oh_16_oh_e1_balance", "ratio")])
        self.assertNotIn("ratio", {r["unit"] for r in rows})
        meta = list(csv.DictReader(open(self.dir / "dutch_default_metadata.csv")))
        self.assertEqual(meta, [{"record_id": "rd001", "group": "PSSD", "tier": "C", "drug": "sertraline"}])

    def test_no_export_carries_an_id_url_quote_image_or_author_key(self):
        text = "".join(p.read_text() for p in self.dir.iterdir())
        for secret in ("secretpostid", "secretcommentid", "hrtpostid", "t3_", "t1_", "5ecre7auth0rkey",
                       "0therauth0rkey1", "http", "redd", "secretimage", "deadbeef", "test cyp"):
            self.assertNotIn(secret, text)

    def test_default_export_loads_in_the_analysis(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
        try:
            from asclepius.dataset import load_lab_reports
        except ImportError:
            self.skipTest("run with analysis/.venv/bin/python to load the export")
        data = load_lab_reports([self.dir / "dutch_default.csv"], metadata=self.dir / "dutch_default_metadata.csv")
        self.assertEqual(list(data.records["group"]), ["PSSD"])
        self.assertEqual(data.unmatched, ())


class FindImages(unittest.TestCase):
    def test_gallery_post_gives_full_size_images_in_gallery_order(self):
        post = {"id": "abc", "name": "t3_abc", "title": "My DUTCH results", "selftext": "", "author": "x",
                "created_utc": 1, "url": "https://www.reddit.com/gallery/abc",
                "gallery_data": {"items": [{"media_id": "m2"}, {"media_id": "m1"}]},
                "media_metadata": {m: {"e": "Image", "s": {"u": f"https://preview.redd.it/{m}.png?width=900&amp;s=z"}}
                                   for m in ("m1", "m2")}}
        self.assertEqual(d.image_urls(post), ["https://preview.redd.it/m2.png?width=900&s=z",
                                              "https://preview.redd.it/m1.png?width=900&s=z"])

    def test_comment_links_and_imgur_pages(self):
        c = {"body": "dutch: https://i.redd.it/q1.jpg and [here](https://imgur.com/AbC12) and https://imgur.com/a/alb"}
        self.assertEqual(d.image_urls(c), ["https://i.redd.it/q1.jpg", "https://i.imgur.com/AbC12.jpg",
                                           "https://imgur.com/a/alb"])

    def test_item_needs_dutch_and_an_image(self):
        salt = b"s" * 32
        self.assertIsNone(d.dutch_item({"body": "my dutch was low", "author": "x"}, salt))
        self.assertIsNone(d.dutch_item({"body": "Dutchess https://i.redd.it/q.jpg", "author": "x"}, salt))
        it = d.dutch_item({"body": "DUTCH https://i.redd.it/q.jpg", "author": "x", "name": "t1_c",
                           "created_utc": 5, "link_id": "t3_p"}, salt)
        self.assertEqual((it["id"], it["urls"], it["author_key"]), ("t1_c", ["https://i.redd.it/q.jpg"],
                                                                     d.author_key("x", salt)))
        self.assertIsNone(d.dutch_item({"body": "DUTCH https://i.redd.it/q.jpg", "author": "[deleted]"}, salt))


class ReviewSample(unittest.TestCase):
    def test_a_saved_sample_is_reused_so_checked_reports_never_change(self):
        path = Path(tempfile.mkdtemp()) / "sample.json"
        reports = [report(f"t3_{i}") for i in range(30)]
        first, _ = d.review_sample(reports, 5, seed=0, path=path)
        again, _ = d.review_sample([report("t3_new")] + reports, 5, seed=1, path=path)
        self.assertEqual([r["item_id"] for r in again], [r["item_id"] for r in first])


class Score(unittest.TestCase):
    def test_wrong_and_missed_markers_count_as_errors(self):
        rows = [{"correct": "y"}] * 47 + [{"correct": "n"}, {"correct": "missed"}, {"correct": ""}]
        s = d.score(rows)
        self.assertEqual((s["checked"], s["unmarked"]), (49, 1))
        self.assertAlmostEqual(s["error_rate"], 2 / 49)


if __name__ == "__main__":
    unittest.main()
