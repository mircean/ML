"""
Simple test for load_portfolio and save_portfolio functions
"""

import agent

# Load portfolio
print("Loading portfolio...")
portfolio = agent.load_portfolio()

print(f"Cash: ${portfolio['cash']:.2f}")
print(f"Positions: {len(portfolio['positions'])}")
print(f"Closed lots: {len(portfolio.get('closed_lots', {}))}")
print(f"Total value: ${portfolio.get('total_value', 0):.2f}")
if "prices_as_of" in portfolio:
    print(f"Prices as of: {portfolio['prices_as_of']}")

# Save portfolio
print("\nSaving portfolio...")
agent.save_portfolio(portfolio)

print("\n✅ Test passed!")
