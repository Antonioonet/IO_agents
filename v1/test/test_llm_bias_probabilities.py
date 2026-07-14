import unittest
from unittest.mock import patch

import llm_bias_probabilities


class EstimateLlmBiasProbabilitiesTests(unittest.TestCase):
    def test_applies_laplace_smoothing_to_four_action_counts(self):
        decisions = [
            "post",
            "post",
            "post",
            "post",
            "reply",
            "reply",
            "reply",
            "retweet",
            "retweet",
            "do_nothing",
        ]

        with patch.object(
            llm_bias_probabilities,
            "call_constrained_action",
            side_effect=decisions,
        ):
            probabilities = (
                llm_bias_probabilities.estimate_llm_bias_probabilities(
                    persona="Test persona",
                    feed_pool=[f"tweet {index}" for index in range(20)],
                    prior_samples=10,
                    feed_size=10,
                    prior_seed=123,
                )
            )

        self.assertEqual(probabilities["llm_bias_post_prob"], 5 / 14)
        self.assertEqual(probabilities["llm_bias_reply_prob"], 4 / 14)
        self.assertEqual(probabilities["llm_bias_retweet_prob"], 3 / 14)
        self.assertEqual(probabilities["llm_bias_do_nothing_prob"], 2 / 14)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_feed_snapshots_are_reproducible_and_have_ten_tweets(self):
        feed_pool = [f"tweet {index}" for index in range(30)]

        def collect_prompts():
            prompts = []

            def choose_post(prompt, **_):
                prompts.append(prompt)
                return "post"

            with patch.object(
                llm_bias_probabilities,
                "call_constrained_action",
                side_effect=choose_post,
            ):
                llm_bias_probabilities.estimate_llm_bias_probabilities(
                    persona="Test persona",
                    feed_pool=feed_pool,
                    prior_samples=10,
                    feed_size=10,
                    prior_seed=456,
                )
            return prompts

        first_prompts = collect_prompts()
        second_prompts = collect_prompts()

        self.assertEqual(first_prompts, second_prompts)
        self.assertEqual(len(first_prompts), 10)
        for prompt in first_prompts:
            feed_lines = [line for line in prompt.splitlines() if line.startswith("[")]
            self.assertEqual(len(feed_lines), 10)


class CollectLlmChoiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_collects_choice_from_oasis_feed_without_agent_action(self):
        feed_posts = [
            {"content": "First tweet", "username": "alice"},
            {"content": "Second tweet"},
            "Third tweet",
        ]

        with patch.object(
            llm_bias_probabilities,
            "call_constrained_action",
            return_value="reply",
        ) as constrained_action:
            choice = await llm_bias_probabilities.collect_llm_choice(
                persona="Test persona",
                feed_posts=feed_posts,
                model="test-model",
                ollama_url="http://localhost:11434",
                request_timeout=30,
                seed=42,
            )

        self.assertEqual(choice, "reply")
        prompt = constrained_action.call_args.args[0]
        self.assertIn("Test persona", prompt)
        self.assertIn("[1] @alice: First tweet", prompt)
        self.assertIn("[2] Second tweet", prompt)
        self.assertIn("[3] Third tweet", prompt)
        self.assertEqual(
            constrained_action.call_args.kwargs,
            {
                "model": "test-model",
                "ollama_url": "http://localhost:11434",
                "request_timeout": 30,
                "seed": 42,
            },
        )

    async def test_rejects_a_feed_without_text(self):
        with self.assertRaisesRegex(ValueError, "empty feed"):
            await llm_bias_probabilities.collect_llm_choice(
                persona="Test persona",
                feed_posts=[{"content": ""}, None],
            )


if __name__ == "__main__":
    unittest.main()
