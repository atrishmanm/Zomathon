"""
Generate synthetic datasets for Zomato KPT prediction problem
This script creates realistic datasets simulating:
- Order data with true KPT and merchant-marked FOR
- Merchant characteristics
- Rider assignment and wait times
- Kitchen rush patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

def generate_merchant_data(n_merchants=1000):
    """Generate merchant characteristics dataset"""
    
    merchant_types = ['QSR', 'Fine Dining', 'Cloud Kitchen', 'Cafe', 'Street Food']
    city_tiers = ['Tier 1', 'Tier 2', 'Tier 3']
    
    merchants = {
        'merchant_id': [f'M{str(i).zfill(6)}' for i in range(1, n_merchants + 1)],
        'merchant_type': np.random.choice(merchant_types, n_merchants, 
                                         p=[0.35, 0.15, 0.25, 0.15, 0.10]),
        'city_tier': np.random.choice(city_tiers, n_merchants, 
                                     p=[0.40, 0.35, 0.25]),
        'avg_monthly_orders': np.random.lognormal(6, 1.5, n_merchants).astype(int),
        'kitchen_capacity': np.random.randint(5, 50, n_merchants),
        'has_multiple_platforms': np.random.choice([True, False], n_merchants, p=[0.75, 0.25]),
        'has_dine_in': np.random.choice([True, False], n_merchants, p=[0.60, 0.40]),
        'marking_reliability_score': np.random.beta(5, 2, n_merchants),  # Most are reliable
        'avg_menu_prep_complexity': np.random.uniform(1, 5, n_merchants),
    }
    
    df = pd.DataFrame(merchants)
    return df

def generate_order_data(merchants_df, n_orders=50000):
    """Generate order dataset with KPT signals"""
    
    start_date = datetime(2026, 1, 1)
    
    orders = []
    
    for order_id in range(1, n_orders + 1):
        # Select random merchant
        merchant = merchants_df.sample(1).iloc[0]
        
        # Generate order timestamp
        days_offset = np.random.randint(0, 60)
        # Hour distribution (0-23) representing order frequency throughout day
        # Low activity: 0-5 (night), High activity: 12-14, 19-21 (lunch/dinner)
        hour_weights = np.array([1]*6 + [2]*3 + [8]*3 + [12]*3 + [6]*3 + [8]*3 + [2]*3)
        hour_probs = hour_weights / hour_weights.sum()
        hour = int(np.random.choice(range(24), p=hour_probs))
        minute = np.random.randint(0, 60)
        order_time = start_date + timedelta(days=int(days_offset), 
                                           hours=hour, 
                                           minutes=minute)
        
        # True KPT based on merchant characteristics and rush
        base_kpt = 15 + merchant['avg_menu_prep_complexity'] * 5
        
        # Rush hour factor (lunch and dinner)
        if hour in [12, 13, 14, 19, 20, 21]:
            rush_factor = np.random.uniform(1.3, 1.8)
        elif hour in [11, 15, 18, 22]:
            rush_factor = np.random.uniform(1.1, 1.3)
        else:
            rush_factor = np.random.uniform(0.8, 1.1)
        
        # External load (dine-in + other platforms)
        external_load_factor = 1.0
        if merchant['has_dine_in'] and hour in [12, 13, 19, 20]:
            external_load_factor *= np.random.uniform(1.2, 1.6)
        if merchant['has_multiple_platforms']:
            external_load_factor *= np.random.uniform(1.1, 1.4)
        
        # True KPT
        true_kpt = base_kpt * rush_factor * external_load_factor
        true_kpt = max(5, min(90, true_kpt + np.random.normal(0, 3)))
        
        # Merchant-marked FOR time (with bias)
        reliability = merchant['marking_reliability_score']
        
        if reliability > 0.8:
            # Reliable merchants: close to true KPT
            marked_kpt = true_kpt + np.random.normal(0, 2)
        elif reliability > 0.6:
            # Moderately reliable: some bias
            marked_kpt = true_kpt + np.random.normal(2, 4)
        else:
            # Unreliable: significant bias (rider-influenced)
            # Sometimes mark when rider arrives
            rider_arrival_time = true_kpt + np.random.uniform(5, 15)
            if np.random.random() < 0.4:
                marked_kpt = rider_arrival_time
            else:
                marked_kpt = true_kpt + np.random.normal(5, 6)
        
        marked_kpt = max(3, marked_kpt)
        
        # Order characteristics
        order_value = np.random.lognormal(4, 0.8)
        num_items = np.random.poisson(2.5) + 1
        
        # Current model prediction (based on noisy historical data)
        # Has systematic error due to biased training labels
        predicted_kpt = marked_kpt * 0.93 + np.random.normal(0, 5)
        predicted_kpt = max(5, predicted_kpt)
        
        # Rider assignment time (based on predicted KPT)
        rider_assigned_at = predicted_kpt * 0.7  # Assign rider to arrive slightly early
        
        # Rider arrival time
        rider_travel_time = np.random.uniform(8, 20)
        rider_arrival_time = rider_assigned_at + rider_travel_time
        
        # Rider wait time (key metric)
        rider_wait_time = max(0, true_kpt - rider_arrival_time)
        early_arrival = max(0, rider_arrival_time - true_kpt)
        
        # Customer ETA error
        customer_eta_predicted = predicted_kpt + rider_travel_time + 5
        customer_eta_actual = true_kpt + rider_travel_time + 5
        eta_error = abs(customer_eta_actual - customer_eta_predicted)
        
        orders.append({
            'order_id': f'O{str(order_id).zfill(8)}',
            'merchant_id': merchant['merchant_id'],
            'order_time': order_time,
            'hour': hour,
            'day_of_week': order_time.strftime('%A'),
            'is_rush_hour': hour in [12, 13, 14, 19, 20, 21],
            'order_value': order_value,
            'num_items': num_items,
            'true_kpt_minutes': round(true_kpt, 2),
            'merchant_marked_for_minutes': round(marked_kpt, 2),
            'predicted_kpt_minutes': round(predicted_kpt, 2),
            'rider_assigned_at_minutes': round(rider_assigned_at, 2),
            'rider_travel_time_minutes': round(rider_travel_time, 2),
            'rider_arrival_time_minutes': round(rider_arrival_time, 2),
            'rider_wait_time_minutes': round(rider_wait_time, 2),
            'early_arrival_minutes': round(early_arrival, 2),
            'customer_eta_error_minutes': round(eta_error, 2),
            'marking_bias': round(marked_kpt - true_kpt, 2),
            'prediction_error': round(predicted_kpt - true_kpt, 2),
        })
    
    df = pd.DataFrame(orders)
    return df

def generate_kitchen_rush_data(merchants_df, n_observations=20000):
    """Generate kitchen rush observations (current vs. actual load)"""
    
    observations = []
    
    for obs_id in range(n_observations):
        merchant = merchants_df.sample(1).iloc[0]
        
        timestamp = datetime(2026, 1, 1) + timedelta(
            days=np.random.randint(0, 60),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        
        hour = timestamp.hour
        
        # Zomato orders (observable)
        base_zomato_orders = merchant['avg_monthly_orders'] / 30 / 24
        if hour in [12, 13, 14, 19, 20, 21]:
            zomato_orders = np.random.poisson(base_zomato_orders * 5)
        else:
            zomato_orders = np.random.poisson(base_zomato_orders * 2)
        
        # Other platform orders (not observable)
        if merchant['has_multiple_platforms']:
            other_platform_orders = np.random.poisson(zomato_orders * 0.8)
        else:
            other_platform_orders = 0
        
        # Dine-in orders (not observable)
        if merchant['has_dine_in'] and hour in [12, 13, 19, 20]:
            dine_in_orders = np.random.poisson(zomato_orders * 1.2)
        else:
            dine_in_orders = 0
        
        total_kitchen_load = zomato_orders + other_platform_orders + dine_in_orders
        
        # Kitchen utilization
        utilization = min(1.0, total_kitchen_load / merchant['kitchen_capacity'])
        
        observations.append({
            'observation_id': f'OBS{str(obs_id).zfill(8)}',
            'merchant_id': merchant['merchant_id'],
            'timestamp': timestamp,
            'hour': hour,
            'zomato_orders_count': zomato_orders,
            'other_platform_orders_count': other_platform_orders,
            'dine_in_orders_count': dine_in_orders,
            'total_kitchen_load': total_kitchen_load,
            'kitchen_capacity': merchant['kitchen_capacity'],
            'utilization_rate': round(utilization, 3),
            'observable_load_ratio': round(zomato_orders / max(1, total_kitchen_load), 3)
        })
    
    df = pd.DataFrame(observations)
    return df

def generate_iot_sensor_data(merchants_df, n_readings=10000):
    """Generate simulated IoT sensor data (proposed solution)"""
    
    # Simulate data for subset of merchants with IoT sensors
    merchants_with_iot = merchants_df.sample(frac=0.20)  # 20% pilot
    
    readings = []
    
    for reading_id in range(n_readings):
        merchant = merchants_with_iot.sample(1).iloc[0]
        
        timestamp = datetime(2026, 1, 1) + timedelta(
            days=np.random.randint(0, 60),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60),
            seconds=np.random.randint(0, 60)
        )
        
        hour = timestamp.hour
        
        # Kitchen activity signals
        # 1. Stove/burner activity
        if hour in [12, 13, 14, 19, 20, 21]:
            burners_active = np.random.randint(3, merchant['kitchen_capacity'])
        else:
            burners_active = np.random.randint(0, max(3, merchant['kitchen_capacity'] // 2))
        
        # 2. Temperature sensors
        ambient_temp = np.random.normal(32, 5)  # Celsius
        
        # 3. Motion/activity sensors
        activity_level = np.random.uniform(0, 1) * (burners_active / merchant['kitchen_capacity'])
        
        # 4. Order preparation stations occupied
        stations_occupied = min(burners_active + np.random.randint(-1, 2), 
                               merchant['kitchen_capacity'])
        
        # 5. Estimated current load
        estimated_load = burners_active + stations_occupied // 2
        
        readings.append({
            'reading_id': f'IOT{str(reading_id).zfill(8)}',
            'merchant_id': merchant['merchant_id'],
            'timestamp': timestamp,
            'hour': hour,
            'burners_active': burners_active,
            'ambient_temperature': round(ambient_temp, 1),
            'activity_level': round(activity_level, 3),
            'stations_occupied': stations_occupied,
            'estimated_current_load': estimated_load,
            'kitchen_capacity': merchant['kitchen_capacity'],
            'estimated_utilization': round(estimated_load / merchant['kitchen_capacity'], 3)
        })
    
    df = pd.DataFrame(readings)
    return df

def main():
    """Generate all datasets"""
    print("Generating synthetic datasets for Zomato KPT prediction...")
    
    print("\n1. Generating merchant data...")
    merchants_df = generate_merchant_data(n_merchants=1000)
    merchants_df.to_csv('data/merchants.csv', index=False)
    print(f"   [OK] Generated {len(merchants_df)} merchants")
    
    print("\n2. Generating order data...")
    orders_df = generate_order_data(merchants_df, n_orders=50000)
    orders_df.to_csv('data/orders.csv', index=False)
    print(f"   [OK] Generated {len(orders_df)} orders")
    
    print("\n3. Generating kitchen rush observations...")
    rush_df = generate_kitchen_rush_data(merchants_df, n_observations=20000)
    rush_df.to_csv('data/kitchen_rush.csv', index=False)
    print(f"   [OK] Generated {len(rush_df)} rush observations")
    
    print("\n4. Generating IoT sensor data...")
    iot_df = generate_iot_sensor_data(merchants_df, n_readings=10000)
    iot_df.to_csv('data/iot_sensors.csv', index=False)
    print(f"   [OK] Generated {len(iot_df)} IoT readings")
    
    print("\n" + "="*60)
    print("Data generation complete!")
    print("="*60)
    
    # Print summary statistics
    print("\n[Summary] Dataset Summary:")
    print(f"   Total Merchants: {len(merchants_df)}")
    print(f"   Total Orders: {len(orders_df)}")
    print(f"   Date Range: {orders_df['order_time'].min()} to {orders_df['order_time'].max()}")
    print(f"   Avg Rider Wait Time: {orders_df['rider_wait_time_minutes'].mean():.2f} minutes")
    print(f"   Avg ETA Error: {orders_df['customer_eta_error_minutes'].mean():.2f} minutes")
    print(f"   Avg Marking Bias: {orders_df['marking_bias'].mean():.2f} minutes")
    print(f"   Merchants with IoT: {len(iot_df['merchant_id'].unique())}")

if __name__ == "__main__":
    main()
