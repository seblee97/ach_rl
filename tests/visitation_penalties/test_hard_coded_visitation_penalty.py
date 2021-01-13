from visitation_penalties import hard_coded_visitation_penalty


class TestHardCodedPenalty:

    HARD_CODED_PENALTY_SEQUENCE = [
        [0, 4.0],
        [8, -2.0],
        [40, 20.0],
        [60, 8.0],
        [100, 0.0],
        [110, 30.0],
    ]

    def test_call_sequence(self):
        penalty_class = hard_coded_visitation_penalty.HardCodedPenalty(
            hard_coded_penalties=self.HARD_CODED_PENALTY_SEQUENCE
        )

        penalty = penalty_class._current_penalty
        observed_penalties = []

        for i in range(150):
            p = penalty_class(episode=i)
            if p != penalty:
                observed_penalties.append([i, p])
                penalty = p

        assert observed_penalties == self.HARD_CODED_PENALTY_SEQUENCE
