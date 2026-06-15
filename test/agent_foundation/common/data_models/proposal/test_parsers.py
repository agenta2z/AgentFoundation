"""Commit 3 (D3): pluggable proposal-parser registry."""
import pytest

from agent_foundation.common.data_models.proposal.parsers import (
    ProposalParser,
    get_proposal_parser,
    register_proposal_parser,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    register_proposal_parser(None)
    yield
    register_proposal_parser(None)


class _Parser:
    def parse(self, workspace: str):
        return {"workspace": workspace}


def test_none_when_unregistered():
    assert get_proposal_parser() is None


def test_register_round_trip():
    p = _Parser()
    register_proposal_parser(p)
    assert get_proposal_parser() is p
    assert get_proposal_parser().parse("/ws") == {"workspace": "/ws"}


def test_replacement_last_registration_wins():
    p1, p2 = _Parser(), _Parser()
    register_proposal_parser(p1)
    register_proposal_parser(p2)
    assert get_proposal_parser() is p2


def test_clear_with_none():
    register_proposal_parser(_Parser())
    register_proposal_parser(None)
    assert get_proposal_parser() is None


def test_parser_satisfies_protocol():
    assert isinstance(_Parser(), ProposalParser)


def test_reexported_from_package():
    from agent_foundation.common.data_models.proposal import (
        ProposalParser as PP,
        get_proposal_parser as gpp,
        register_proposal_parser as rpp,
    )

    assert PP is ProposalParser
    assert gpp is get_proposal_parser
    assert rpp is register_proposal_parser
