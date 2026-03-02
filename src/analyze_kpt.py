"""
Analyze KPT prediction accuracy and identify key issues
"""

import pandas as pd
import numpy as np
import os

def load_data():
    """Load all generated datasets"""
    merchants = pd.read_csv('data/merchants.csv')
    orders = pd.read_csv('data/orders.csv')
    kitchen_rush = pd.read_csv('data/kitchen_rush.csv')
    iot_sensors = pd.read_csv('data/iot_sensors.csv')
    
    return merchants, orders, kitchen_rush, iot_sensors

def analyze_marking_bias(orders):
    """Analyze bias in merchant-marked FOR signals"""
    print("\n" + "="*60)
    print("ANALYSIS 1: Merchant Marking Bias")
    print("="*60)
    
    # Overall bias
    avg_bias = orders['marking_bias'].mean()
    median_bias = orders['marking_bias'].median()
    std_bias = orders['marking_bias'].std()
    
    print(f"\n📊 Overall Marking Bias Statistics:")
    print(f"   Mean Bias: {avg_bias:.2f} minutes")
    print(f"   Median Bias: {median_bias:.2f} minutes")
    print(f"   Std Dev: {std_bias:.2f} minutes")
    
    # Positive vs negative bias
    overestimated = (orders['marking_bias'] > 2).sum()
    underestimated = (orders['marking_bias'] < -2).sum()
    accurate = len(orders) - overestimated - underestimated
    
    print(f"\n📈 Bias Distribution:")
    print(f"   Overestimated (>2 min late marking): {overestimated} ({overestimated/len(orders)*100:.1f}%)")
    print(f"   Accurate (±2 min): {accurate} ({accurate/len(orders)*100:.1f}%)")
    print(f"   Underestimated (<2 min early marking): {underestimated} ({underestimated/len(orders)*100:.1f}%)")
    
    # Rush hour impact
    rush_bias = orders[orders['is_rush_hour']]['marking_bias'].mean()
    non_rush_bias = orders[~orders['is_rush_hour']]['marking_bias'].mean()
    
    print(f"\n⏰ Rush Hour Impact:")
    print(f"   Rush Hour Bias: {rush_bias:.2f} minutes")
    print(f"   Non-Rush Hour Bias: {non_rush_bias:.2f} minutes")
    print(f"   Difference: {abs(rush_bias - non_rush_bias):.2f} minutes")
    
    return {
        'mean_bias': avg_bias,
        'median_bias': median_bias,
        'std_bias': std_bias,
        'rush_bias': rush_bias,
        'non_rush_bias': non_rush_bias
    }

def analyze_prediction_errors(orders):
    """Analyze prediction errors and their impact"""
    print("\n" + "="*60)
    print("ANALYSIS 2: Prediction Errors & Impact")
    print("="*60)
    
    # Prediction errors
    mae = orders['prediction_error'].abs().mean()
    rmse = np.sqrt((orders['prediction_error'] ** 2).mean())
    p50_error = orders['customer_eta_error_minutes'].quantile(0.50)
    p90_error = orders['customer_eta_error_minutes'].quantile(0.90)
    
    print(f"\n📉 Prediction Error Metrics:")
    print(f"   Mean Absolute Error (MAE): {mae:.2f} minutes")
    print(f"   Root Mean Square Error (RMSE): {rmse:.2f} minutes")
    print(f"   P50 ETA Error: {p50_error:.2f} minutes")
    print(f"   P90 ETA Error: {p90_error:.2f} minutes")
    
    # Rider wait times
    avg_wait = orders['rider_wait_time_minutes'].mean()
    avg_early = orders['early_arrival_minutes'].mean()
    total_wasted = orders['rider_wait_time_minutes'].sum() + orders['early_arrival_minutes'].sum()
    
    print(f"\n🏍️ Rider Efficiency Impact:")
    print(f"   Avg Rider Wait Time: {avg_wait:.2f} minutes")
    print(f"   Avg Early Arrival (idle): {avg_early:.2f} minutes")
    print(f"   Total Wasted Rider Time: {total_wasted:,.0f} minutes")
    print(f"   Equivalent to: {total_wasted/60:,.0f} rider-hours wasted")
    
    # Orders with significant delays
    delayed_orders = (orders['rider_wait_time_minutes'] > 10).sum()
    very_delayed = (orders['rider_wait_time_minutes'] > 15).sum()
    
    print(f"\n⚠️ Problematic Orders:")
    print(f"   Orders with >10 min wait: {delayed_orders} ({delayed_orders/len(orders)*100:.1f}%)")
    print(f"   Orders with >15 min wait: {very_delayed} ({very_delayed/len(orders)*100:.1f}%)")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'p50_error': p50_error,
        'p90_error': p90_error,
        'avg_wait': avg_wait,
        'total_wasted': total_wasted
    }

def analyze_kitchen_rush(kitchen_rush):
    """Analyze unobservable kitchen load"""
    print("\n" + "="*60)
    print("ANALYSIS 3: Unobservable Kitchen Load")
    print("="*60)
    
    # Observable vs total load
    avg_observable_ratio = kitchen_rush['observable_load_ratio'].mean()
    median_utilization = kitchen_rush['utilization_rate'].median()
    
    # Rush hour analysis
    rush_hours = kitchen_rush[kitchen_rush['hour'].isin([12, 13, 14, 19, 20, 21])]
    rush_observable_ratio = rush_hours['observable_load_ratio'].mean()
    
    print(f"\n👁️ Visibility Gap:")
    print(f"   Avg % of Kitchen Load Visible to Zomato: {avg_observable_ratio*100:.1f}%")
    print(f"   During Rush Hours: {rush_observable_ratio*100:.1f}%")
    print(f"   Invisible Load: {(1-avg_observable_ratio)*100:.1f}%")
    
    print(f"\n🔥 Kitchen Utilization:")
    print(f"   Median Utilization Rate: {median_utilization*100:.1f}%")
    print(f"   High Utilization (>80%): {(kitchen_rush['utilization_rate'] > 0.8).sum()} observations")
    
    # Breakdown by source
    total_zomato = kitchen_rush['zomato_orders_count'].sum()
    total_other = kitchen_rush['other_platform_orders_count'].sum()
    total_dine_in = kitchen_rush['dine_in_orders_count'].sum()
    total_all = total_zomato + total_other + total_dine_in
    
    print(f"\n📦 Order Source Breakdown:")
    print(f"   Zomato Orders: {total_zomato:,} ({total_zomato/total_all*100:.1f}%)")
    print(f"   Other Platforms: {total_other:,} ({total_other/total_all*100:.1f}%)")
    print(f"   Dine-In: {total_dine_in:,} ({total_dine_in/total_all*100:.1f}%)")
    
    return {
        'observable_ratio': avg_observable_ratio,
        'rush_observable_ratio': rush_observable_ratio,
        'median_utilization': median_utilization
    }

def calculate_business_impact(orders):
    """Calculate business impact of current issues"""
    print("\n" + "="*60)
    print("ANALYSIS 4: Business Impact (Scaled to Zomato)")
    print("="*60)
    
    # Scale to Zomato's actual volume
    # Assume ~1.5M orders per day across India
    daily_orders = 1_500_000
    sample_orders = len(orders)
    scale_factor = daily_orders / sample_orders
    
    # Costs
    avg_wait = orders['rider_wait_time_minutes'].mean()
    avg_early = orders['early_arrival_minutes'].mean()
    
    # Per order impact
    wasted_time_per_order = avg_wait + avg_early
    total_daily_waste = wasted_time_per_order * daily_orders
    
    # Assuming rider time costs ~₹5 per minute (wages + opportunity cost)
    cost_per_minute = 5
    daily_cost = total_daily_waste * cost_per_minute
    monthly_cost = daily_cost * 30
    
    print(f"\n💰 Cost Impact:")
    print(f"   Wasted Time per Order: {wasted_time_per_order:.2f} minutes")
    print(f"   Total Daily Wasted Time: {total_daily_waste:,.0f} minutes ({total_daily_waste/60:,.0f} hours)")
    print(f"   Daily Cost: ₹{daily_cost:,.0f}")
    print(f"   Monthly Cost: ₹{monthly_cost:,.0f}")
    print(f"   Annual Cost: ₹{monthly_cost*12:,.0f}")
    
    # Customer experience
    eta_errors = orders['customer_eta_error_minutes']
    high_error_pct = (eta_errors > 10).sum() / len(orders) * 100
    
    print(f"\n😊 Customer Experience:")
    print(f"   Orders with >10min ETA error: {high_error_pct:.1f}%")
    print(f"   Scaled Daily Impact: ~{high_error_pct/100 * daily_orders:,.0f} customers affected")
    
    # Potential cancellations (assume 2% cancel when wait >15 min)
    very_delayed = (orders['rider_wait_time_minutes'] > 15).sum() / len(orders)
    potential_cancellations = very_delayed * 0.02 * daily_orders
    avg_order_value = 350  # ₹
    lost_gmv = potential_cancellations * avg_order_value
    
    print(f"\n❌ Order Cancellation Risk:")
    print(f"   Potential Daily Cancellations: ~{potential_cancellations:.0f}")
    print(f"   Estimated Daily GMV Loss: ₹{lost_gmv:,.0f}")
    print(f"   Estimated Monthly GMV Loss: ₹{lost_gmv*30:,.0f}")
    
    return {
        'daily_cost': daily_cost,
        'monthly_cost': monthly_cost,
        'daily_gmv_loss': lost_gmv,
        'orders_affected_pct': high_error_pct
    }

def main():
    """Run all analyses"""
    print("\n" + "="*80)
    print(" "*20 + "ZOMATO KPT PREDICTION ANALYSIS")
    print("="*80)
    
    # Load data
    print("\n[Loading] Loading datasets...")
    merchants, orders, kitchen_rush, iot_sensors = load_data()
    print(f"   [OK] Loaded: {len(merchants)} merchants, {len(orders)} orders")
    
    # Run analyses
    bias_results = analyze_marking_bias(orders)
    prediction_results = analyze_prediction_errors(orders)
    rush_results = analyze_kitchen_rush(kitchen_rush)
    business_results = calculate_business_impact(orders)
    
    # Summary
    print("\n" + "="*80)
    print("🎯 KEY FINDINGS SUMMARY")
    print("="*80)
    print("\n1. MARKING BIAS PROBLEM:")
    print(f"   - Merchant markings are {abs(bias_results['mean_bias']):.1f} minutes biased on average")
    print(f"   - Rush hour bias is {abs(bias_results['rush_bias'] - bias_results['non_rush_bias']):.1f}min worse")
    
    print("\n2. VISIBILITY GAP:")
    print(f"   - Only {rush_results['observable_ratio']*100:.0f}% of kitchen load is visible")
    print(f"   - {(1-rush_results['observable_ratio'])*100:.0f}% of kitchen activity is unobservable")
    
    print("\n3. PREDICTION ACCURACY:")
    print(f"   - P90 ETA Error: {prediction_results['p90_error']:.1f} minutes")
    print(f"   - Average rider wait: {prediction_results['avg_wait']:.1f} minutes")
    
    print("\n4. BUSINESS IMPACT:")
    print(f"   - Monthly cost: ₹{business_results['monthly_cost']:,.0f}")
    print(f"   - Potential monthly GMV loss: ₹{business_results['daily_gmv_loss']*30:,.0f}")
    
    print("\n" + "="*80)
    print("✅ Analysis complete! Check images/ folder for visualizations.")
    print("="*80)

if __name__ == "__main__":
    main()
