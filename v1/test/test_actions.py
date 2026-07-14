import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd

import actions


def persona_row():
    return {
        "name": "test-user",
        "do_nothing_prob": 0.4,
        "create_post_prob": 0.3,
        "create_comment_prob": 0.18,
        "repost_prob": 0.12,
        "llm_bias_do_nothing_prob": 0.1,
        "llm_bias_post_prob": 0.5,
        "llm_bias_reply_prob": 0.3,
        "llm_bias_retweet_prob": 0.1,
    }


class CorrectedActionProbabilitiesTests(unittest.TestCase):
    def test_applies_logit_correction_and_softmax(self):
        row = pd.Series(persona_row())

        probabilities = actions.corrected_action_probabilities(
            row,
            llm_choice="reply",
            beta=1.0,
        )

        grounded = {
            "do_nothing": 0.4,
            "post": 0.3,
            "reply": 0.18,
            "retweet": 0.12,
        }
        bias = {
            "do_nothing": 0.1,
            "post": 0.5,
            "reply": 0.3,
            "retweet": 0.1,
        }
        expected_weights = {
            action: math.exp(
                (1.0 if action == "reply" else 0.0)
                - math.log(bias[action])
                + math.log(grounded[action])
            )
            for action in actions.CALIBRATED_ACTIONS
        }
        expected_total = sum(expected_weights.values())

        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        for action in actions.CALIBRATED_ACTIONS:
            self.assertAlmostEqual(
                probabilities[action],
                expected_weights[action] / expected_total,
            )


class LogiticPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_collects_raw_choice_and_returns_selected_tool_prompt(self):
        profiles = pd.DataFrame([persona_row()])
        user_info = SimpleNamespace(
            name="test-user",
            description="Test persona",
            profile="Test persona",
        )

        with (
            patch.object(
                actions,
                "collect_llm_choice",
                new=AsyncMock(return_value="post"),
            ) as collect_choice,
            patch.object(
                actions,
                "corrected_action_probabilities",
                return_value={
                    "do_nothing": 0.0,
                    "post": 0.0,
                    "reply": 1.0,
                    "retweet": 0.0,
                },
            ),
            patch.object(actions.random, "choices", return_value=["reply"]),
        ):
            prompt = await actions.logitic_prompt(
                user_info,
                profiles,
                feed_posts=[{"content": "A feed tweet"}],
                llm_seed=456,
            )

        self.assertIn("- create_comment", prompt)
        collect_choice.assert_awaited_once()
        self.assertEqual(
            collect_choice.await_args.kwargs["feed_posts"],
            [{"content": "A feed tweet"}],
        )
        self.assertEqual(collect_choice.await_args.kwargs["seed"], 456)


class SetTextPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_constructs_custom_oasis_profile_as_a_mapping(self):
        profile = persona_row()
        profile.update(
            {
                "username": "test-user",
                "user_char": "Test persona",
                "description": "Test persona",
                "I.O": False,
            }
        )
        original_user_info = actions.UserInfo(
            user_name="test-user",
            name="test-user",
            description="Test persona",
            profile={"other_info": {"user_profile": "Test persona"}},
            recsys_type="twitter",
            is_controllable=True,
        )
        original_agent = SimpleNamespace(
            user_info=original_user_info,
            social_agent_id=0,
            channel=object(),
        )
        agent_graph = SimpleNamespace(
            get_agents=lambda: [(0, original_agent)],
            agent_mappings={},
        )
        args = SimpleNamespace(action_mode="natural")

        def construct_agent(**kwargs):
            rendered_prompt = kwargs["user_info"].to_custom_system_message(
                kwargs["user_info_template"]
            )
            self.assertIn("Test persona", rendered_prompt)
            return SimpleNamespace(**kwargs)

        with tempfile.TemporaryDirectory() as temp_dir:
            profile_path = Path(temp_dir) / "personas.csv"
            pd.DataFrame([profile]).to_csv(profile_path, index=False)
            with patch.object(
                actions,
                "SocialAgent",
                side_effect=construct_agent,
            ):
                await actions.set_text_prompt(
                    args=args,
                    agent_graph=agent_graph,
                    profile_path=profile_path,
                    model=object(),
                    available_actions=[],
                )

        replacement = agent_graph.agent_mappings[0]
        self.assertEqual(
            replacement.user_info.profile,
            {"description": "Test persona"},
        )
        self.assertEqual(replacement.user_info.user_name, "test-user")
        self.assertTrue(replacement.user_info.is_controllable)


if __name__ == "__main__":
    unittest.main()
