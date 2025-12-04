"""
Monte Carlo Simulation Utility
Implements Geometric Brownian Motion for stock price forecasting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from typing import Tuple, List, Dict, Optional
from pathlib import Path


class MonteCarloSimulator:
    """Monte Carlo simulation for stock price prediction"""
    
    def __init__(self, num_simulations: int = 1000, random_seed: int = 42):
        """
        Initialize Monte Carlo simulator
        
        Args:
            num_simulations: Number of simulation paths
            random_seed: Random seed for reproducibility
        """
        self.num_simulations = num_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def calculate_parameters(
        self,
        price_data: pd.Series,
        method: str = 'historical'
    ) -> Tuple[float, float]:
        """
        Calculate drift and volatility parameters
        
        Args:
            price_data: Historical price series
            method: Calculation method ('historical' or 'exponential')
            
        Returns:
            Tuple of (drift, volatility)
        """
        # Calculate daily returns
        returns = price_data.pct_change().dropna()
        
        if method == 'historical':
            # Historical mean and std
            drift = returns.mean()
            volatility = returns.std()
        elif method == 'exponential':
            # Exponentially weighted
            drift = returns.ewm(span=20).mean().iloc[-1]
            volatility = returns.ewm(span=20).std().iloc[-1]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Annualized parameters
        drift_annual = drift * 252
        volatility_annual = volatility * np.sqrt(252)
        
        print(f"✓ Calculated parameters using {method} method:")
        print(f"  Daily drift: {drift:.6f} ({drift_annual:.4f} annualized)")
        print(f"  Daily volatility: {volatility:.6f} ({volatility_annual:.4f} annualized)")
        
        return drift, volatility
    
    def simulate_gbm(
        self,
        S0: float,
        drift: float,
        volatility: float,
        days: int,
        num_sims: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate stock prices using Geometric Brownian Motion
        
        Formula: S(t) = S(0) * exp((μ - 0.5σ²)t + σ√t * Z)
        Where:
            S(t) = price at time t
            S(0) = initial price
            μ = drift (expected return)
            σ = volatility (standard deviation)
            Z = random variable from standard normal distribution
        
        Args:
            S0: Initial stock price
            drift: Expected return (daily)
            volatility: Standard deviation (daily)
            days: Number of days to simulate
            num_sims: Number of simulations (default: self.num_simulations)
            
        Returns:
            Array of shape (num_sims, days+1) with simulated prices
        """
        if num_sims is None:
            num_sims = self.num_simulations
        
        # Time steps
        dt = 1  # Daily
        
        # Initialize price matrix
        prices = np.zeros((num_sims, days + 1))
        prices[:, 0] = S0
        
        # Generate random shocks
        Z = np.random.standard_normal((num_sims, days))
        
        # Calculate price paths using GBM formula
        for t in range(1, days + 1):
            prices[:, t] = prices[:, t-1] * np.exp(
                (drift - 0.5 * volatility**2) * dt +
                volatility * np.sqrt(dt) * Z[:, t-1]
            )
        
        return prices
    
    def run_simulation(
        self,
        price_history: pd.Series,
        time_horizons: List[int] = [30, 60, 90],
        method: str = 'historical'
    ) -> Dict:
        """
        Run complete Monte Carlo simulation
        
        Args:
            price_history: Historical price data
            time_horizons: List of days to forecast
            method: Parameter calculation method
            
        Returns:
            Dictionary with simulation results for each horizon
        """
        # Get initial price
        S0 = price_history.iloc[-1]
        
        print(f"\n{'='*60}")
        print("MONTE CARLO SIMULATION")
        print(f"{'='*60}")
        print(f"Initial Price: ${S0:.2f}")
        print(f"Simulations: {self.num_simulations:,}")
        print(f"Horizons: {time_horizons} days")
        
        # Calculate parameters
        drift, volatility = self.calculate_parameters(price_history, method)
        
        # Run simulations for each horizon
        results = {}
        
        for horizon in time_horizons:
            print(f"\nSimulating {horizon}-day forecast...")
            
            # Run simulation
            prices = self.simulate_gbm(S0, drift, volatility, horizon)
            
            # Final prices distribution
            final_prices = prices[:, -1]
            
            # Calculate statistics
            stats = self.calculate_statistics(final_prices, S0, horizon)
            
            # Store results
            results[f'{horizon}d'] = {
                'horizon': horizon,
                'paths': prices,
                'final_prices': final_prices,
                'statistics': stats
            }
            
            # Print summary
            self.print_summary(stats, horizon)
        
        print(f"{'='*60}\n")
        
        return results
    
    def calculate_statistics(
        self,
        final_prices: np.ndarray,
        S0: float,
        horizon: int
    ) -> Dict:
        """
        Calculate statistical measures from simulated prices
        
        Args:
            final_prices: Array of final prices from simulations
            S0: Initial price
            horizon: Forecast horizon in days
            
        Returns:
            Dictionary with statistical measures
        """
        # Calculate returns
        returns = (final_prices - S0) / S0 * 100
        
        # Percentiles for confidence intervals
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        price_percentiles = np.percentile(final_prices, percentiles)
        
        # Probability of profit
        prob_profit = (final_prices > S0).sum() / len(final_prices) * 100
        
        # Expected return
        expected_return = returns.mean()
        
        # Value at Risk (VaR) - 5% worst case
        var_5 = np.percentile(returns, 5)
        
        # Confidence intervals (68% and 95%)
        ci_68 = (np.percentile(final_prices, 16), np.percentile(final_prices, 84))
        ci_95 = (np.percentile(final_prices, 2.5), np.percentile(final_prices, 97.5))
        
        stats = {
            'mean': float(final_prices.mean()),
            'median': float(np.median(final_prices)),
            'std': float(final_prices.std()),
            'min': float(final_prices.min()),
            'max': float(final_prices.max()),
            'percentiles': {
                f'p{p}': float(v) for p, v in zip(percentiles, price_percentiles)
            },
            'expected_return_pct': float(expected_return),
            'prob_profit_pct': float(prob_profit),
            'var_5_pct': float(var_5),
            'ci_68': (float(ci_68[0]), float(ci_68[1])),
            'ci_95': (float(ci_95[0]), float(ci_95[1])),
            'initial_price': float(S0),
            'horizon_days': horizon
        }
        
        return stats
    
    def print_summary(self, stats: Dict, horizon: int):
        """Print simulation summary statistics"""
        print(f"\n  {horizon}-Day Forecast Summary:")
        print(f"    Expected Price:    ${stats['mean']:.2f}")
        print(f"    Median Price:      ${stats['median']:.2f}")
        print(f"    Price Range:       ${stats['min']:.2f} - ${stats['max']:.2f}")
        print(f"    Expected Return:   {stats['expected_return_pct']:.2f}%")
        print(f"    Prob. of Profit:   {stats['prob_profit_pct']:.1f}%")
        print(f"    VaR (5%):          {stats['var_5_pct']:.2f}%")
        print(f"    68% CI:            ${stats['ci_68'][0]:.2f} - ${stats['ci_68'][1]:.2f}")
        print(f"    95% CI:            ${stats['ci_95'][0]:.2f} - ${stats['ci_95'][1]:.2f}")
    
    def plot_simulation_results(
        self,
        results: Dict,
        price_history: pd.Series,
        show_paths: int = 100,
        figsize: Tuple[int, int] = (16, 12)
    ):
        """
        Plot simulation results
        
        Args:
            results: Results dictionary from run_simulation()
            price_history: Historical price data
            show_paths: Number of paths to display
            figsize: Figure size
        """
        num_horizons = len(results)
        fig, axes = plt.subplots(num_horizons, 2, figsize=figsize)
        
        if num_horizons == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (key, data) in enumerate(results.items()):
            horizon = data['horizon']
            paths = data['paths']
            final_prices = data['final_prices']
            stats = data['statistics']
            
            # Left plot: Price paths
            ax_paths = axes[idx, 0]
            
            # Plot sample paths
            time_steps = np.arange(horizon + 1)
            for i in range(min(show_paths, self.num_simulations)):
                ax_paths.plot(time_steps, paths[i, :], alpha=0.3, linewidth=0.5, color='blue')
            
            # Plot mean path
            mean_path = paths.mean(axis=0)
            ax_paths.plot(time_steps, mean_path, color='red', linewidth=2, label='Mean Path')
            
            # Plot confidence intervals
            ci_95_lower = np.percentile(paths, 2.5, axis=0)
            ci_95_upper = np.percentile(paths, 97.5, axis=0)
            ax_paths.fill_between(time_steps, ci_95_lower, ci_95_upper, alpha=0.2, color='green', label='95% CI')
            
            ax_paths.set_title(f'{horizon}-Day Price Paths (showing {min(show_paths, self.num_simulations)} of {self.num_simulations})')
            ax_paths.set_xlabel('Days')
            ax_paths.set_ylabel('Price ($)')
            ax_paths.legend()
            ax_paths.grid(True, alpha=0.3)
            
            # Right plot: Final price distribution
            ax_dist = axes[idx, 1]
            
            ax_dist.hist(final_prices, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax_dist.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: ${stats['mean']:.2f}")
            ax_dist.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: ${stats['median']:.2f}")
            ax_dist.axvline(stats['initial_price'], color='black', linestyle='-', linewidth=2, label=f"Initial: ${stats['initial_price']:.2f}")
            
            # Add percentile lines
            ax_dist.axvline(stats['percentiles']['p5'], color='orange', linestyle=':', linewidth=1, alpha=0.7)
            ax_dist.axvline(stats['percentiles']['p95'], color='orange', linestyle=':', linewidth=1, alpha=0.7)
            
            ax_dist.set_title(f'{horizon}-Day Final Price Distribution')
            ax_dist.set_xlabel('Price ($)')
            ax_dist.set_ylabel('Frequency')
            ax_dist.legend()
            ax_dist.grid(True, alpha=0.3)
            
            # Add text with statistics
            textstr = f'Expected Return: {stats["expected_return_pct"]:.2f}%\n'
            textstr += f'Prob. Profit: {stats["prob_profit_pct"]:.1f}%\n'
            textstr += f'VaR (5%): {stats["var_5_pct"]:.2f}%'
            ax_dist.text(0.02, 0.98, textstr, transform=ax_dist.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        # Save to data directory using Path object
        output_dir = Path(__file__).parent.parent / 'data'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'monte_carlo_results.png')
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved as '{output_path}'")
        plt.show()
    
    def create_summary_table(self, results: Dict) -> pd.DataFrame:
        """
        Create summary table of simulation results
        
        Args:
            results: Results dictionary from run_simulation()
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for key, data in results.items():
            stats = data['statistics']
            
            summary_data.append({
                'Horizon': f"{stats['horizon_days']} days",
                'Initial Price': f"${stats['initial_price']:.2f}",
                'Expected Price': f"${stats['mean']:.2f}",
                'Median Price': f"${stats['median']:.2f}",
                'Expected Return': f"{stats['expected_return_pct']:.2f}%",
                'Probability of Profit': f"{stats['prob_profit_pct']:.1f}%",
                'VaR (5%)': f"{stats['var_5_pct']:.2f}%",
                '95% CI Lower': f"${stats['ci_95'][0]:.2f}",
                '95% CI Upper': f"${stats['ci_95'][1]:.2f}",
                'Price Range': f"${stats['min']:.2f} - ${stats['max']:.2f}"
            })
        
        df = pd.DataFrame(summary_data)
        
        return df
