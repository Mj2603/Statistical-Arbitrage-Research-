from dataclasses import dataclass


@dataclass
class TransactionCostModel:
    brokerage_bps: float = 3.0
    half_spread_bps: float = 5.0
    slippage_bps: float = 2.0

    def round_trip_cost(self, notional: float) -> float:
        total_bps = 2 * (self.brokerage_bps + self.half_spread_bps + self.slippage_bps)
        return notional * total_bps / 10_000

    def single_leg_cost(self, notional: float) -> float:
        total_bps = self.brokerage_bps + self.half_spread_bps + self.slippage_bps
        return notional * total_bps / 10_000
