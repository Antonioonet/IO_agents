import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import persona_generation


class GeneratePersonasTests(unittest.TestCase):
    def test_writes_llm_bias_probability_columns_to_csv(self):
        normal_rows = pd.DataFrame(
            {
                "userid": [1] * 10,
                "tweet_text": [f"user tweet {index}" for index in range(10)],
                "tweet_type": ["post"] * 10,
                "user_display_name": ["Real Name"] * 10,
                "user_screen_name": ["real_name"] * 10,
            }
        )
        io_rows = normal_rows.iloc[0:0].copy()

        def fake_ollama(_prompt, **_):
            fake_ollama.unconstrained_calls += 1
            return "fake_user" if fake_ollama.unconstrained_calls == 1 else "A persona"

        fake_ollama.unconstrained_calls = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            normal_path = temp_dir / "normal.pkl"
            io_path = temp_dir / "io.pkl"
            output_path = temp_dir / "personas.csv"
            normal_rows.to_pickle(normal_path)
            io_rows.to_pickle(io_path)

            bias_probabilities = {
                "llm_bias_do_nothing_prob": 1 / 14,
                "llm_bias_post_prob": 11 / 14,
                "llm_bias_reply_prob": 1 / 14,
                "llm_bias_retweet_prob": 1 / 14,
            }
            with (
                patch.object(
                    persona_generation,
                    "call_ollama",
                    side_effect=fake_ollama,
                ),
                patch.object(
                    persona_generation,
                    "estimate_llm_bias_probabilities",
                    return_value=bias_probabilities,
                ),
            ):
                generated = persona_generation.generate_personas(
                    normal_file=normal_path,
                    io_file=io_path,
                    normal_limit=1,
                    io_limit=0,
                    prior_samples=10,
                    prior_feed_size=10,
                    prior_seed=789,
                    output_path=output_path,
                )

            written = pd.read_csv(output_path)

        expected_columns = {
            "llm_bias_do_nothing_prob",
            "llm_bias_post_prob",
            "llm_bias_reply_prob",
            "llm_bias_retweet_prob",
        }
        self.assertTrue(expected_columns.issubset(generated.columns))
        self.assertTrue(expected_columns.issubset(written.columns))
        self.assertAlmostEqual(
            generated.loc[0, list(expected_columns)].sum(),
            1.0,
        )
        self.assertEqual(generated.loc[0, "llm_bias_post_prob"], 11 / 14)


if __name__ == "__main__":
    unittest.main()
