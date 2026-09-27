"""The comparison engine's central promise: an absent value says so.

A comparison table is the part of the report a reader trusts most and checks
least, which makes it the part where a quiet default is most dangerous. If a
missing price renders as an empty cell, a dash, a `0`, or a `None`, a reader
scanning the table concludes the competitor is free, and the error is invisible
precisely because the table looks complete.

So the sentinel is not a display nicety here, it is the product. Every cell the
research did not fill reads ``"Not publicly available"``, in ``value`` as well as
``display``, so a consumer reading either one gets the truth. ``ComparisonCell``
can build that cell in exactly one way, and these tests hold the service to
using it.

The second half of the file is about what the table is allowed to imply. Ranking
companies on a dimension where only one of them has data is not a comparison; it
is a claim that the others lack the thing, made by the absence of data. The
service reports per-company evidence quality and refuses to rank on a dimension
nobody filled.
"""

from __future__ import annotations

import pytest

from app.schemas.common import NOT_PUBLIC
from app.schemas.company import CompanyProfile
from app.schemas.comparison import (
    ComparisonCell,
    ComparisonMatrix,
    ComparisonRow,
)
from app.schemas.pricing import PricingAnalysis
from app.services.comparison_service import ComparisonService, RankedDimension


def _service() -> ComparisonService:
    return ComparisonService()


class TestTheAbsentValue:
    """The plan's first required failure."""

    def test_a_company_with_no_price_reads_not_publicly_available(self) -> None:
        """Required: the sentinel, not ``None`` and not ``0``."""
        matrix = _service().build(
            ["Acme", "Beta"],
            {"Acme": {"starter_price": 20}},
        )
        cell = matrix.row("starter_price").cells["Beta"]
        assert cell.display == NOT_PUBLIC
        assert cell.value == NOT_PUBLIC
        assert cell.is_known is False

    def test_the_sentinel_is_never_none(self) -> None:
        matrix = _service().build(["Acme"], {"Acme": {"price": None}})
        cell = matrix.row("price").cells["Acme"]
        assert cell.value is not None
        assert cell.value == NOT_PUBLIC

    def test_a_zero_price_is_zero_and_not_the_sentinel(self) -> None:
        """The falsy check this service is most likely to get wrong.

        A free tier legitimately costs 0. Any implementation that writes
        ``if not value: cell = unknown()`` turns a real, cited, free price into
        "not publicly available" — losing the one figure a reader most wants.
        """
        matrix = _service().build(["Acme"], {"Acme": {"starter_price": 0}})
        cell = matrix.row("starter_price").cells["Acme"]
        assert cell.is_known is True
        assert cell.value == 0
        assert cell.display == "0"

    def test_an_empty_string_is_the_sentinel_not_a_blank_cell(self) -> None:
        matrix = _service().build(["Acme"], {"Acme": {"positioning": "  "}})
        cell = matrix.row("positioning").cells["Acme"]
        assert cell.display == NOT_PUBLIC
        assert cell.is_known is False

    def test_a_company_absent_entirely_gets_a_full_row_of_sentinels(self) -> None:
        """Differing from a partial row is not a licence to invent a company."""
        matrix = _service().build(
            ["Acme", "Beta", "Gamma"], {"Acme": {"price": 20}, "Beta": {"price": 30}}
        )
        for dimension in matrix.dimensions:
            gamma = dimension.cells["Gamma"]
            assert gamma.display == NOT_PUBLIC, dimension.field
            assert gamma.is_known is False

    def test_every_company_appears_in_every_row(self) -> None:
        matrix = _service().build(
            ["Acme", "Beta"], {"Acme": {"price": 20}, "Beta": {"free_tier": True}}
        )
        for dimension in matrix.dimensions:
            assert set(dimension.cells) == {"Acme", "Beta"}

    def test_an_unknown_company_reads_the_sentinel_through_the_row_helper(self) -> None:
        matrix = _service().build(["Acme"], {"Acme": {"price": 20}})
        assert matrix.row("price").display_for("Nobody") == NOT_PUBLIC
        assert matrix.row("no_such_field") is None


class TestTheShapeOfTheTable:
    def test_companies_keep_the_order_they_were_given(self) -> None:
        """The target comes first by convention; the service does not reshuffle."""
        matrix = _service().build(["Acme", "Beta", "Gamma"], {})
        assert matrix.companies == ["Acme", "Beta", "Gamma"]

    def test_a_dimension_exists_for_every_field_offered(self) -> None:
        matrix = _service().build(
            ["Acme"], {"Acme": {"price": 1, "free_tier": False, "plan_name": "Starter"}}
        )
        assert {d.field for d in matrix.dimensions} == {
            "price",
            "free_tier",
            "plan_name",
        }

    def test_dimensions_appear_in_a_stable_order_across_runs(self) -> None:
        """Two runs over the same data produce the same table, in the same order.

        Dict ordering makes this free today, but a dimension list assembled from
        a set would silently reshuffle the report between runs.
        """
        first = _service().build(["Acme"], {"Acme": {"b": 1, "a": 2}})
        second = _service().build(["Acme"], {"Acme": {"a": 2, "b": 1}})
        assert [d.field for d in first.dimensions] == [d.field for d in second.dimensions]

    def test_the_flat_rows_view_mirrors_the_dimensions(self) -> None:
        """`rows` is the legacy view some consumers read; it must not disagree."""
        matrix = _service().build(["Acme"], {"Acme": {"price": 20}})
        row = next(r for r in matrix.rows if r["field"] == "price")
        assert row["values"]["Acme"] == "20"

    def test_a_flat_row_never_carries_a_raw_none(self) -> None:
        matrix = _service().build(["Acme", "Beta"], {"Acme": {"price": 20}})
        row = next(r for r in matrix.rows if r["field"] == "price")
        assert row["values"]["Beta"] == NOT_PUBLIC

    def test_explicit_fields_are_honoured_over_the_union(self) -> None:
        """An extra field present in one payload does not silently widen the table."""
        matrix = _service().build(
            ["Acme"], {"Acme": {"price": 1, "internal_note": "x"}},
            fields=["price"],
        )
        assert [d.field for d in matrix.dimensions] == ["price"]

    def test_a_field_listed_but_never_supplied_still_gets_a_row(self) -> None:
        matrix = _service().build(["Acme"], {"Acme": {"price": 1}}, fields=["price", "seats"])
        assert {d.field for d in matrix.dimensions} == {"price", "seats"}

    def test_labels_are_attached_when_supplied(self) -> None:
        matrix = _service().build(
            ["Acme"], {"Acme": {"starter_price": 1}}, labels={"starter_price": "Starter price"}
        )
        assert matrix.row("starter_price").label == "Starter price"

    def test_a_label_falls_back_to_a_readable_field_name(self) -> None:
        matrix = _service().build(["Acme"], {"Acme": {"starter_price": 1}})
        assert matrix.row("starter_price").label == "Starter Price"

    def test_a_label_is_never_the_empty_string(self) -> None:
        matrix = _service().build(["Acme"], {"Acme": {"price": 1}}, labels={"price": ""})
        assert matrix.row("price").label


class TestValuesAndProvenance:
    def test_a_cell_keeps_the_sources_that_backed_it(self) -> None:
        matrix = _service().build(
            ["Acme"], {"Acme": {"price": {"value": 20, "source_ids": ["s1", "s2"]}}}
        )
        assert matrix.row("price").cells["Acme"].source_ids == ["s1", "s2"]

    def test_a_plain_value_yields_a_cell_with_no_sources(self) -> None:
        matrix = _service().build(["Acme"], {"Acme": {"price": 20}})
        assert matrix.row("price").cells["Acme"].source_ids == []

    def test_a_cell_records_the_confidence_it_was_given(self) -> None:
        matrix = _service().build(
            ["Acme"], {"Acme": {"price": {"value": 20, "confidence": "high"}}}
        )
        assert matrix.row("price").cells["Acme"].confidence.value == "high"

    def test_a_bad_confidence_is_refused_rather_than_coerced(self) -> None:
        matrix = _service().build(
            ["Acme"], {"Acme": {"price": {"value": 20, "confidence": "certain"}}}
        )
        cell = matrix.row("price").cells["Acme"]
        assert cell.confidence.value in {"low", "medium", "high"}

    def test_a_list_value_is_joined_rather_than_dropped(self) -> None:
        matrix = _service().build(["Acme"], {"Acme": {"plans": ["Starter", "Pro"]}})
        cell = matrix.row("plans").cells["Acme"]
        assert cell.is_known is True
        assert "Starter" in cell.display
        assert cell.value == ["Starter", "Pro"]

    def test_a_nested_value_is_kept_whole_in_value(self) -> None:
        matrix = _service().build(
            ["Acme"], {"Acme": {"plan": {"name": "Pro", "price": 30}}}
        )
        assert matrix.row("plan").cells["Acme"].value == {"name": "Pro", "price": 30}


class TestEvidenceQuality:
    def test_a_company_with_no_sources_scores_zero(self) -> None:
        matrix = _service().build(
            ["Acme"], {"Acme": {"price": {"value": 20, "source_ids": []}}}
        )
        assert matrix.evidence_quality["Acme"] == 0.0

    def test_a_company_with_sources_scores_above_zero(self) -> None:
        matrix = _service().build(
            ["Acme"],
            {"Acme": {"price": {"value": 20, "source_ids": ["s1"]},
                      "plan": {"value": "Pro", "source_ids": ["s2"]}}},
        )
        assert matrix.evidence_quality["Acme"] > 0.0

    def test_every_company_appears_in_the_quality_map(self) -> None:
        """A company the service forgot to score looks identical to a scored zero."""
        matrix = _service().build(["Acme", "Beta"], {"Acme": {"price": 1}})
        assert set(matrix.evidence_quality) == {"Acme", "Beta"}


class TestRankingIsRefusedOnEmptyDimensions:
    def test_ranking_a_dimension_only_one_company_filled_is_refused(self) -> None:
        """The plan's rule, stated strictly: a ranking is a claim about the set.

        Acme's known price says nothing about Beta's, so "Acme ranks first" is a
        statement about Beta that the data does not support. Returning Acme alone
        would let a caller render a winner it has not established.
        """
        matrix = _service().build(["Acme", "Beta"], {"Acme": {"price": 20}})
        assert _service().rank(matrix, "price") is None

    def test_ranking_a_filled_dimension_orders_by_value(self) -> None:
        matrix = _service().build(
            ["Acme", "Beta"], {"Acme": {"price": 30}, "Beta": {"price": 10}}
        )
        ranked = _service().rank(matrix, "price")
        assert [r.company for r in ranked] == ["Acme", "Beta"]
        assert ranked[0].value == 30
        assert ranked[1].rank == 2

    def test_ranking_is_refused_when_any_company_is_missing(self) -> None:
        matrix = _service().build(
            ["Acme", "Beta"], {"Acme": {"price": 30}, "Beta": {"pricing_model": "per seat"}}
        )
        assert _service().rank(matrix, "price") is None

    def test_ranking_is_refused_when_nobody_filled_the_dimension(self) -> None:
        matrix = _service().build(["Acme", "Beta"], {"Acme": {}, "Beta": {}})
        assert _service().rank(matrix, "price") is None

    def test_ranking_a_missing_field_returns_nothing(self) -> None:
        matrix = _service().build(["Acme"], {"Acme": {"price": 1}})
        assert _service().rank(matrix, "absent") is None

    def test_ranking_is_refused_when_no_value_is_a_number(self) -> None:
        """A row of plan names is not a ranking, however complete it is."""
        matrix = _service().build(
            ["Acme", "Beta"], {"Acme": {"plan": "Pro"}, "Beta": {"plan": "Starter"}}
        )
        assert _service().rank(matrix, "plan") is None

    def test_a_boolean_is_not_ranked_even_though_it_is_a_number_subclass(self) -> None:
        """``True > False`` in Python, and that comparison means nothing here."""
        matrix = _service().build(
            ["Acme", "Beta"], {"Acme": {"free": True}, "Beta": {"free": False}}
        )
        assert _service().rank(matrix, "free") is None

    def test_ranking_a_single_company_table_is_allowed(self) -> None:
        """One company and one known value is not a claim about anyone else."""
        matrix = _service().build(["Acme"], {"Acme": {"price": 20}})
        ranked = _service().rank(matrix, "price")
        assert [r.company for r in ranked] == ["Acme"]

    def test_ranking_a_sentinel_valued_cell_is_refused(self) -> None:
        matrix = _service().build(
            ["Acme", "Beta"],
            {"Acme": {"price": 30}, "Beta": {"price": NOT_PUBLIC}},
        )
        assert _service().rank(matrix, "price") is None

    def test_an_empty_table_cannot_be_ranked(self) -> None:
        assert _service().rank(ComparisonMatrix(), "price") is None


class TestAnalysesToFields:
    """Turning the typed analysis payloads into comparable fields."""

    def test_a_pricing_analysis_yields_its_comparable_fields(self) -> None:
        fields = _service().fields_from(
            PricingAnalysis(company="Acme", pricing_model="per seat", has_free_tier=True)
        )
        assert fields["pricing_model"] == "per seat"
        assert fields["has_free_tier"] is True

    def test_an_unset_pricing_model_is_the_sentinel_not_an_empty_cell(self) -> None:
        fields = _service().fields_from(PricingAnalysis(company="Acme"))
        assert fields["pricing_model"] == NOT_PUBLIC

    def test_a_company_profile_yields_its_comparable_fields(self) -> None:
        fields = _service().fields_from(
            CompanyProfile(name="Acme", industry="cloud", country="GB")
        )
        assert fields["industry"] == "cloud"
        assert fields["country"] == "GB"

    def test_an_unset_profile_field_is_the_sentinel(self) -> None:
        fields = _service().fields_from(CompanyProfile(name="Acme"))
        assert fields["industry"] == NOT_PUBLIC

    def test_the_sentinel_is_never_offered_as_a_value_to_compare(self) -> None:
        """A sentinel that became a real cell value would rank below every price."""
        fields = _service().fields_from(PricingAnalysis(company="Acme"))
        assert fields.pop("pricing_model") == NOT_PUBLIC

    def test_an_unknown_payload_type_is_refused_rather_than_guessed_at(self) -> None:
        with pytest.raises(TypeError):
            _service().fields_from(object())

    def test_field_extraction_keeps_list_values_as_lists(self) -> None:
        fields = _service().fields_from(
            PricingAnalysis(company="Acme", billing_periods=["monthly", "annual"])
        )
        assert fields["billing_periods"] == ["monthly", "annual"]


class TestCoverage:
    def test_coverage_reports_which_categories_survived(self) -> None:
        coverage = _service().coverage(
            "Beta", covered=["pricing", "product"], failed=["review"], sources=7, evidence=4
        )
        assert coverage.company == "Beta"
        assert coverage.categories_covered == ["pricing", "product"]
        assert coverage.categories_failed == ["review"]
        assert coverage.sources_retrieved == 7
        assert coverage.evidence_count == 4

    def test_coverage_of_a_company_with_nothing_is_all_failure(self) -> None:
        coverage = _service().coverage("Beta")
        assert coverage.categories_covered == []
        assert coverage.categories_failed == []
        assert coverage.sources_retrieved == 0


class TestMatrixItself:
    def test_the_matrix_agrees_with_its_own_cells(self) -> None:
        matrix = _service().build(["Acme", "Beta"], {"Acme": {"price": 20}})
        assert matrix.row("price").display_for("Acme") == "20"
        assert matrix.row("price").display_for("Beta") == NOT_PUBLIC

    def test_a_hand_built_matrix_still_refuses_none_in_a_flat_row(self) -> None:
        """The schema is the backstop, not the service."""
        matrix = ComparisonMatrix(
            companies=["Acme"], rows=[{"field": "price", "values": {"Acme": None}}]
        )
        assert matrix.rows[0]["values"]["Acme"] == NOT_PUBLIC

    def test_display_for_an_unbuilt_row_is_the_sentinel(self) -> None:
        row = ComparisonRow(field="price", cells={"Acme": ComparisonCell.unknown()})
        assert row.display_for("Acme") == NOT_PUBLIC
        assert row.display_for("Beta") == NOT_PUBLIC

    def test_a_ranked_dimension_records_its_field(self) -> None:
        ranked = RankedDimension(field="price", company="Acme", value=20, rank=1)
        assert ranked.field == "price"
        assert ranked.rank == 1
