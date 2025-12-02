"""
Portfolio Performance Chart

Creates a chart comparing portfolio performance vs NASDAQ 100 index.
Shows normalized percentage returns from the starting date.
"""

import logging
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from portfolio_database import PortfolioDatabase

logger = logging.getLogger(__name__)


def create_performance_chart(days: int = 365, save_path: str = "portfolio_performance.png"):
    """
    Create a chart comparing portfolio vs NASDAQ 100 performance.

    Args:
        days: Number of days of history to fetch
        save_path: Path to save the chart image
    """
    # Fetch data
    logger.info(f"Fetching {days} days of portfolio and NASDAQ 100 history...")
    db = PortfolioDatabase()
    portfolio_history = db.get_portfolio_history(days=days)
    nasdaq_history = db.get_nasdaq100_history(days=days)

    if not portfolio_history or not nasdaq_history:
        logger.error("No historical data available")
        return

    # Convert to DataFrames
    portfolio_df = pd.DataFrame(portfolio_history)
    portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
    portfolio_df = portfolio_df.set_index('date')

    nasdaq_df = pd.DataFrame(nasdaq_history)
    nasdaq_df['date'] = pd.to_datetime(nasdaq_df['date'])
    nasdaq_df = nasdaq_df.set_index('date')

    # Merge on date
    df = portfolio_df[['total_value']].join(nasdaq_df[['value']], how='inner')
    df.columns = ['Portfolio', 'NASDAQ100']

    # Calculate normalized returns (percentage change from start)
    df_normalized = (df / df.iloc[0] * 100) - 100  # Percentage change from start

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Portfolio Performance vs NASDAQ 100', fontsize=16, fontweight='bold')

    # Plot 1: Normalized Returns (%)
    ax1.plot(df_normalized.index, df_normalized['Portfolio'],
             label='Portfolio', linewidth=2, marker='o', color='#2E86AB')
    ax1.plot(df_normalized.index, df_normalized['NASDAQ100'],
             label='NASDAQ 100', linewidth=2, marker='s', color='#A23B72')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Return (%)', fontsize=12)
    ax1.set_title('Normalized Returns (% Change from Start)', fontsize=13)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Add performance metrics
    portfolio_return = df_normalized['Portfolio'].iloc[-1]
    nasdaq_return = df_normalized['NASDAQ100'].iloc[-1]
    outperformance = portfolio_return - nasdaq_return

    metrics_text = f'Portfolio: {portfolio_return:+.2f}%  |  NASDAQ 100: {nasdaq_return:+.2f}%  |  Outperformance: {outperformance:+.2f}%'
    ax1.text(0.5, 0.98, metrics_text, transform=ax1.transAxes,
             ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Absolute Values (dual y-axis)
    ax2_nasdaq = ax2.twinx()

    line1 = ax2.plot(df.index, df['Portfolio'],
                     label='Portfolio Value ($)', linewidth=2, marker='o', color='#2E86AB')
    line2 = ax2_nasdaq.plot(df.index, df['NASDAQ100'],
                            label='NASDAQ 100 Index', linewidth=2, marker='s', color='#A23B72')

    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Portfolio Value ($)', fontsize=12, color='#2E86AB')
    ax2_nasdaq.set_ylabel('NASDAQ 100 Index', fontsize=12, color='#A23B72')
    ax2.set_title('Absolute Values', fontsize=13)
    ax2.tick_params(axis='y', labelcolor='#2E86AB')
    ax2_nasdaq.tick_params(axis='y', labelcolor='#A23B72')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='best')

    # Add current values
    current_portfolio = df['Portfolio'].iloc[-1]
    current_nasdaq = df['NASDAQ100'].iloc[-1]
    values_text = f'Current Portfolio: ${current_portfolio:,.2f}  |  NASDAQ 100: {current_nasdaq:,.2f}'
    ax2.text(0.5, 0.98, values_text, transform=ax2.transAxes,
             ha='center', va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Format dates on x-axis
    fig.autofmt_xdate()

    # Adjust layout
    plt.tight_layout()

    # Save chart
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Chart saved to {save_path}")

    # Display chart
    plt.show()

    # Print summary statistics
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Period: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} days)")
    print(f"\nPortfolio:")
    print(f"  Starting Value: ${df['Portfolio'].iloc[0]:,.2f}")
    print(f"  Current Value:  ${df['Portfolio'].iloc[-1]:,.2f}")
    print(f"  Return:         {portfolio_return:+.2f}%")
    print(f"\nNASDAQ 100:")
    print(f"  Starting Value: {df['NASDAQ100'].iloc[0]:,.2f}")
    print(f"  Current Value:  {df['NASDAQ100'].iloc[-1]:,.2f}")
    print(f"  Return:         {nasdaq_return:+.2f}%")
    print(f"\nOutperformance:   {outperformance:+.2f}%")
    print("="*70)


def main():
    """Main function"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create chart with all available history
    create_performance_chart(days=365)


if __name__ == "__main__":
    main()
