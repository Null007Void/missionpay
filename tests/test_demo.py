from missionpay.demo import run_signature_demo


def test_signature_demo_exposes_policy_separation_and_payment_refusal():
    result = run_signature_demo()

    assert result["garden_fund"]["outcome"] == "ineligible"
    assert "FUND_ACTIVITY_RESTRICTED" in result["garden_fund"]["reasons"]
    assert result["river_cleanup_fund"]["outcome"] == "eligible"
    assert result["unauthenticated_payment"]["result"] == "refused"
    assert result["unauthenticated_payment"]["reason"] == "AUTHENTICATION_REQUIRED"
    assert result["exact_authorized_intent"]["created"] is True
    assert result["exact_authorized_intent"]["amount_cents"] == 7899
