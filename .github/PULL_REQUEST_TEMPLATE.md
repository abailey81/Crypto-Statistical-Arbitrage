## Description

<!-- Provide a concise summary of your changes and the motivation behind them. -->

## Component(s) Affected

<!-- Check all that apply -->

- [ ] Data Collection (Phase 1)
- [ ] Altcoin StatArb Strategy (Phase 2)
- [ ] BTC Futures Curve Trading (Phase 3)
- [ ] Backtesting Engine
- [ ] Portfolio Optimization
- [ ] Reporting / Visualizations
- [ ] Configuration
- [ ] Execution Layer
- [ ] Documentation
- [ ] Tests
- [ ] CI / Infrastructure

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to change)
- [ ] Performance improvement
- [ ] Refactoring (no functional changes)
- [ ] Documentation update
- [ ] New venue integration

## Testing Checklist

- [ ] I have run `make test` and all tests pass
- [ ] I have run `make lint` with no errors
- [ ] I have run `make format` to ensure consistent code style
- [ ] I have added tests that cover my changes (if applicable)
- [ ] I have verified the changes work with Python 3.10, 3.11, and 3.12

## Backtest Validation (if strategy changes)

- [ ] Walk-forward backtest results are consistent with baseline
- [ ] No degradation in Sharpe ratio, max drawdown, or win rate
- [ ] Compliance validator passes (`python run_arb.py --validate`)

## Checklist

- [ ] My code follows the project code style (Black + Flake8)
- [ ] I have updated documentation where needed
- [ ] I have not committed API keys, credentials, or `.env` files
- [ ] My changes do not introduce new dependencies without justification

## Additional Notes

<!-- Any context, screenshots, or performance comparisons that reviewers should know about. -->
