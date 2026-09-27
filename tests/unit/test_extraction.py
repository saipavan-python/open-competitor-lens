"""Turning a retrieved page into schema-shaped data without inventing any.

The service's whole promise is that a field it cannot fill from the page stays
absent, and the report says so with the unknown sentinel rather than a plausible
guess. That promise is only meaningful if a field it *can* fill actually arrives,
which is what the first test class here is about: the version of this code that
read ``response.parsed`` — an attribute ``LLMResponse`` does not have — returned an
empty dict for every successful extraction, because the missing attribute was
swallowed by a ``getattr(..., None) or {}``. The model was called, billed, and its
answer discarded, and the run reported a clean list of failures instead of a
silent success with nothing in it.

The second class covers the failure modes that must not end a run. A run that
analysed nine categories cannot lose all of them to one page whose model response
was unparseable, so a bad response is a recorded error on one result rather than
an exception.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from app.schemas.provider import FetchResult, LLMResponse
from app.services.extraction_service import ExtractionService

PAGE_TEXT = (
    "Acme Cloud charges $99 per month for the Pro plan. The Team plan is $249 per "
    "month and includes SSO. There is no setup fee and no charge for read-only "
    "accounts."
)


class Pricing(BaseModel):
    plan: str
    price: str


class _LLM:
    """An LLM that returns a fixed body, standing in for a real provider."""

    def __init__(self, body: str) -> None:
        self.body = body
        self.calls = 0
        self.schema_seen: object = None

    async def generate(self, prompt: str, schema: object = None) -> LLMResponse:
        self.calls += 1
        self.schema_seen = schema
        return LLMResponse(
            content=self.body, model="test-model", provider="stub", prompt_tokens=1
        )


class _BrokenLLM:
    async def generate(self, prompt: str, schema: object = None) -> LLMResponse:
        raise RuntimeError("the provider is down")


def _page(text: str = PAGE_TEXT) -> FetchResult:
    return FetchResult(
        url="https://acmecloud.com/pricing",
        final_url="https://acmecloud.com/pricing",
        text=text,
        status=200,
        retrieved_at=datetime.now(UTC),
    )


class TestTheModelAnswerArrives:
    async def test_extracted_fields_are_actually_returned(self) -> None:
        """The regression: a successful call must not yield an empty dict.

        ``ExtractionResult.ok`` is defined as "no error and some data", so an
        always-empty ``data`` made every successful extraction look like a
        failure — while the exception-free path hid the real cause.
        """
        llm = _LLM('{"plan": "Pro", "price": "$99 per month"}')
        result = await ExtractionService(llm).extract(_page(), Pricing)

        assert result.error is None
        assert result.data == {"plan": "Pro", "price": "$99 per month"}
        assert result.ok

    async def test_a_pydantic_schema_is_validated_not_returned_raw(self) -> None:
        """The model is held to the schema it was asked for.

        The error names the offending field, so the shortfall in a report can be
        traced to a specific missing value rather than "the model was wrong".
        """
        llm = _LLM('{"plan": "Pro"}')  # no price: fails validation
        result = await ExtractionService(llm).extract(_page(), Pricing)

        assert result.error is not None
        assert "price" in result.error
        assert not result.ok

    async def test_the_requested_schema_is_passed_to_the_provider(self) -> None:
        llm = _LLM('{"plan": "Pro", "price": "$99"}')
        await ExtractionService(llm).extract(_page(), Pricing)
        assert llm.schema_seen is Pricing

    async def test_a_quote_is_taken_from_the_page_not_the_model(self) -> None:
        """The quote must be text the reader can find on the page.

        A model can paraphrase a price; the page is what the report cites. So the
        quote is located in the retrieved text, and a value with no matching
        sentence yields an empty quote, which reads as unverified downstream
        rather than as verified with a citation that does not exist.
        """
        llm = _LLM('{"plan": "Team", "price": "$249 per month"}')
        result = await ExtractionService(llm).extract(_page(), Pricing)

        assert result.quote
        assert result.quote in PAGE_TEXT

    async def test_a_value_absent_from_the_page_gets_no_quote(self) -> None:
        """An unverifiable value must not come back looking cited."""
        llm = _LLM('{"plan": "Enterprise", "price": "call us"}')
        result = await ExtractionService(llm).extract(_page(), Pricing)

        assert result.data["plan"] == "Enterprise"
        assert result.quote == ""


class TestUnusableAnswersAreReportedNotRaised:
    async def test_a_fenced_answer_is_still_read(self) -> None:
        """Models emit fences even when told not to."""
        llm = _LLM('```json\n{"plan": "Pro", "price": "$99 per month"}\n```')
        result = await ExtractionService(llm).extract(_page(), Pricing)

        assert result.data["plan"] == "Pro"
        assert result.error is None

    async def test_prose_around_the_json_is_tolerated(self) -> None:
        llm = _LLM(
            'Here is what the page says: {"plan": "Pro", "price": "$99 per month"} '
            "Hope that helps!"
        )
        result = await ExtractionService(llm).extract(_page(), Pricing)

        assert result.data["price"] == "$99 per month"

    async def test_a_non_json_answer_is_an_error_on_that_result(self) -> None:
        llm = _LLM("I could not find pricing on this page.")
        result = await ExtractionService(llm).extract(_page(), Pricing)

        assert result.error
        assert not result.ok
        assert result.data == {}

    async def test_a_provider_outage_is_an_error_not_an_exception(self) -> None:
        """One page's failure must not end the run."""
        result = await ExtractionService(_BrokenLLM()).extract(_page(), Pricing)

        assert result.error
        assert "down" in result.error

    async def test_the_url_is_kept_so_the_failure_can_be_located(self) -> None:
        """A recorded error a reader cannot trace to a page is not actionable."""
        result = await ExtractionService(_BrokenLLM()).extract(_page(), Pricing)

        assert result.url == "https://acmecloud.com/pricing"


class TestNoModelAndNoPage:
    async def test_without_an_llm_the_raw_text_is_returned(self) -> None:
        result = await ExtractionService().extract(_page(), Pricing)

        assert result.data == {"text": PAGE_TEXT}
        assert result.ok

    async def test_a_failed_fetch_never_reaches_the_model(self) -> None:
        llm = _LLM('{"plan": "Pro", "price": "$99"}')
        page = FetchResult(
            url="https://acmecloud.com/pricing",
            status=500,
            retrieved_at=datetime.now(UTC),
            retrieval_status="failed",
            error="HTTP 500",
        )

        result = await ExtractionService(llm).extract(page, Pricing)

        assert result.error == "HTTP 500"
        assert llm.calls == 0
