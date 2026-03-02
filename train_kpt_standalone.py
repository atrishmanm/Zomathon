"""
Self-contained KPT Prediction Model Training Script
Enhanced with 5 new features for improved prediction accuracy
No external imports from codebase - fully self-contained
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Create models directory
os.makedirs('models', exist_ok=True)

def load_and_prepare_data():
    """Load all datasets and prepare features"""
    print("\n" + "="*80)
    print("Loading and Preparing Data")
    print("="*80)
    
    # Load datasets
    merchants = pd.read_csv('data/merchants.csv')
    orders = pd.read_csv('data/orders.csv')
    kitchen_rush = pd.read_csv('data/kitchen_rush.csv')
    iot_sensors = pd.read_csv('data/iot_sensors.csv')
    
    print(f"✓ Loaded {len(orders)} orders from {len(merchants)} merchants")
    
    # Merge datasets
    data = orders.merge(merchants, on='merchant_id', how='left')
    
    # Extract time features from order_time
    data['order_time'] = pd.to_datetime(data['order_time'])
    data['hour'] = data['order_time'].dt.hour
    data['day_of_week'] = data['order_time'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    
    # Merge with kitchen rush data (approximate matching by merchant and hour)
    # For each order, find the corresponding kitchen rush metrics
    kitchen_rush['timestamp'] = pd.to_datetime(kitchen_rush['timestamp'])
    kitchen_rush['hour'] = kitchen_rush['timestamp'].dt.hour
    
    # Aggregate kitchen rush by merchant and hour
    rush_agg = kitchen_rush.groupby(['merchant_id', 'hour']).agg({
        'total_kitchen_load': 'mean',
        'utilization_rate': 'mean',
        'observable_load_ratio': 'mean'
    }).reset_index()
    rush_agg.columns = ['merchant_id', 'hour', 'avg_total_orders', 'avg_utilization', 'avg_observable_ratio']
    
    data = data.merge(rush_agg, on=['merchant_id', 'hour'], how='left')
    
    # Merge with IoT sensor data (aggregate by merchant and hour)
    iot_sensors['timestamp'] = pd.to_datetime(iot_sensors['timestamp'])
    iot_sensors['hour'] = iot_sensors['timestamp'].dt.hour
    
    iot_agg = iot_sensors.groupby(['merchant_id', 'hour']).agg({
        'burners_active': 'mean',
        'ambient_temperature': 'mean',
        'activity_level': 'mean',
        'stations_occupied': 'mean',
        'estimated_current_load': 'mean'
    }).reset_index()
    iot_agg.columns = ['merchant_id', 'hour', 'avg_burners_active', 'avg_temperature', 
                        'avg_activity_level', 'avg_stations_occupied', 'avg_estimated_load']
    
    data = data.merge(iot_agg, on=['merchant_id', 'hour'], how='left')
    
    # Fill missing values with median
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    
    print(f"✓ Prepared {len(data)} records with {len(data.columns)} features")
    
    return data

def create_baseline_features(data):
    """Create baseline features (current system - only merchant marking)"""
    features = [
        'merchant_marked_for_minutes',  # The biased signal we currently have
        'is_rush_hour',
        'hour',
        'day_of_week',
        'is_weekend'
    ]
    
    # Encode categorical variables
    data_encoded = data.copy()
    
    return data_encoded[features]

def create_enhanced_features(data):
    """Create enhanced features with multi-signal approach + 5 NEW FEATURES"""
    
    # Encode categorical variables
    le_type = LabelEncoder()
    le_tier = LabelEncoder()
    
    data_encoded = data.copy()
    data_encoded['merchant_type_encoded'] = le_type.fit_transform(data['merchant_type'])
    data_encoded['city_tier_encoded'] = le_tier.fit_transform(data['city_tier'])
    
    # ============= NEW FEATURES (5 Enhancements) =============
    
    # Feature 1: Order Value to Items Ratio (complexity indicator)
    # Higher ratio suggests more complex/expensive items that may take longer
    data_encoded['order_value_per_item'] = data_encoded['order_value'] / (data_encoded['num_items'] + 1)
    
    # Feature 2: Time Since Last Rush Hour (temporal dynamics)
    # Distance from nearest rush hour affects kitchen state
    peak_hours = [12, 13, 19, 20]  # Lunch and dinner rushes
    data_encoded['dist_to_nearest_rush'] = data_encoded['hour'].apply(
        lambda h: min([abs(h - peak) for peak in peak_hours])
    )
    
    # Feature 3: Kitchen Efficiency Score
    # Ratio of estimated load to capacity (higher = more stressed)
    data_encoded['kitchen_efficiency_score'] = (
        data_encoded['avg_estimated_load'] / (data_encoded['kitchen_capacity'] + 1)
    ).fillna(0)
    
    # Feature 4: Order Complexity Score
    # Combined metric of items and value normalized by merchant averages
    merchant_avg_items = data_encoded.groupby('merchant_id')['num_items'].transform('mean')
    merchant_avg_value = data_encoded.groupby('merchant_id')['order_value'].transform('mean')
    data_encoded['order_complexity_score'] = (
        (data_encoded['num_items'] / (merchant_avg_items + 1)) * 
        (data_encoded['order_value'] / (merchant_avg_value + 1))
    ).fillna(1)
    
    # Feature 5: Merchant Historical Error Bias
    # Track merchant's tendency to over/under estimate
    merchant_avg_bias = data_encoded.groupby('merchant_id')['marking_bias'].transform('mean')
    data_encoded['merchant_historical_bias'] = merchant_avg_bias.fillna(0)
    
    features = [
        # Baseline features
        'merchant_marked_for_minutes',
        'is_rush_hour',
        'hour',
        'day_of_week',
        'is_weekend',
        
        # Merchant characteristics
        'merchant_type_encoded',
        'city_tier_encoded',
        'avg_monthly_orders',
        'kitchen_capacity',
        'has_multiple_platforms',
        'has_dine_in',
        'marking_reliability_score',
        'avg_menu_prep_complexity',
        
        # Kitchen rush signals
        'avg_total_orders',
        'avg_utilization',
        'avg_observable_ratio',
        
        # IoT sensor signals
        'avg_burners_active',
        'avg_temperature',
        'avg_activity_level',
        'avg_stations_occupied',
        'avg_estimated_load',
        
        # ★ NEW FEATURES (5) ★
        'order_value_per_item',
        'dist_to_nearest_rush',
        'kitchen_efficiency_score',
        'order_complexity_score',
        'merchant_historical_bias'
    ]
    
    # Convert boolean to int
    data_encoded['has_multiple_platforms'] = data_encoded['has_multiple_platforms'].astype(int)
    data_encoded['has_dine_in'] = data_encoded['has_dine_in'].astype(int)
    
    return data_encoded[features]

def train_baseline_model(X_train, y_train, X_test, y_test):
    """Train baseline model (simulating current system)"""
    print("\n" + "="*80)
    print("BASELINE MODEL: Current System (Merchant Marking Only)")
    print("="*80)
    
    # Simple model - current approach
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Calculate percentile errors
    errors = np.abs(y_test - y_pred)
    p50 = np.percentile(errors, 50)
    p90 = np.percentile(errors, 90)
    p95 = np.percentile(errors, 95)
    
    print(f"\n📊 Performance Metrics:")
    print(f"   MAE:  {mae:.2f} minutes")
    print(f"   RMSE: {rmse:.2f} minutes")
    print(f"   R²:   {r2:.4f}")
    print(f"\n   P50 Error: {p50:.2f} minutes")
    print(f"   P90 Error: {p90:.2f} minutes")
    print(f"   P95 Error: {p95:.2f} minutes")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 5 Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    joblib.dump(model, 'models/baseline_model.pkl')
    print(f"\n✓ Model saved to models/baseline_model.pkl")
    
    return model, mae, rmse, r2, p90

def train_enhanced_model(X_train, y_train, X_test, y_test):
    """Train enhanced model with multi-signal fusion + 5 NEW FEATURES"""
    print("\n" + "="*80)
    print("ENHANCED MODEL: Multi-Signal Fusion + 5 NEW FEATURES")
    print("="*80)
    
    # Advanced model with more features
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Calculate percentile errors
    errors = np.abs(y_test - y_pred)
    p50 = np.percentile(errors, 50)
    p90 = np.percentile(errors, 90)
    p95 = np.percentile(errors, 95)
    
    print(f"\n📊 Performance Metrics:")
    print(f"   MAE:  {mae:.2f} minutes")
    print(f"   RMSE: {rmse:.2f} minutes")
    print(f"   R²:   {r2:.4f}")
    print(f"\n   P50 Error: {p50:.2f} minutes")
    print(f"   P90 Error: {p90:.2f} minutes")
    print(f"   P95 Error: {p95:.2f} minutes")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔝 Top 10 Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Highlight new features
    print(f"\n⭐ NEW FEATURES Performance:")
    new_features = ['order_value_per_item', 'dist_to_nearest_rush', 
                    'kitchen_efficiency_score', 'order_complexity_score', 
                    'merchant_historical_bias']
    for feature in new_features:
        importance = feature_importance[feature_importance['feature'] == feature]['importance'].values
        if len(importance) > 0:
            rank = feature_importance[feature_importance['feature'] == feature].index[0] + 1
            print(f"   {feature}: {importance[0]:.4f} (Rank #{rank})")
    
    # Save model
    joblib.dump(model, 'models/enhanced_model.pkl')
    print(f"\n✓ Model saved to models/enhanced_model.pkl")
    
    return model, mae, rmse, r2, p90, feature_importance

def compare_models(baseline_metrics, enhanced_metrics):
    """Compare baseline vs enhanced model performance"""
    print("\n" + "="*80)
    print("MODEL COMPARISON & IMPROVEMENT")
    print("="*80)
    
    baseline_mae, baseline_rmse, baseline_r2, baseline_p90 = baseline_metrics
    enhanced_mae, enhanced_rmse, enhanced_r2, enhanced_p90 = enhanced_metrics
    
    # Calculate improvements
    mae_improvement = (baseline_mae - enhanced_mae) / baseline_mae * 100
    rmse_improvement = (baseline_rmse - enhanced_rmse) / baseline_rmse * 100
    r2_improvement = (enhanced_r2 - baseline_r2) / baseline_r2 * 100
    p90_improvement = (baseline_p90 - enhanced_p90) / baseline_p90 * 100
    
    print(f"\n📈 Improvement Summary:")
    print(f"\n   Metric                 Baseline    Enhanced    Improvement")
    print(f"   {'-'*60}")
    print(f"   MAE (minutes)          {baseline_mae:>8.2f}    {enhanced_mae:>8.2f}    {mae_improvement:>6.1f}%")
    print(f"   RMSE (minutes)         {baseline_rmse:>8.2f}    {enhanced_rmse:>8.2f}    {rmse_improvement:>6.1f}%")
    print(f"   R² Score               {baseline_r2:>8.4f}    {enhanced_r2:>8.4f}    {r2_improvement:>6.1f}%")
    print(f"   P90 Error (minutes)    {baseline_p90:>8.2f}    {enhanced_p90:>8.2f}    {p90_improvement:>6.1f}%")
    
    # Business impact
    print(f"\n💼 Business Impact (scaled to 1.5M daily orders):")
    
    # Assuming each minute of error costs ₹5 in rider time
    cost_per_minute = 5
    daily_orders = 1_500_000
    
    baseline_daily_cost = baseline_mae * daily_orders * cost_per_minute
    enhanced_daily_cost = enhanced_mae * daily_orders * cost_per_minute
    daily_savings = baseline_daily_cost - enhanced_daily_cost
    monthly_savings = daily_savings * 30
    annual_savings = daily_savings * 365
    
    print(f"   Daily Savings:   ₹{daily_savings:,.0f}")
    print(f"   Monthly Savings: ₹{monthly_savings:,.0f}")
    print(f"   Annual Savings:  ₹{annual_savings:,.0f}")
    
    # Additional impacts
    print(f"\n🎯 Operational Improvements:")
    print(f"   - Rider wait time reduction: ~{mae_improvement:.1f}%")
    print(f"   - ETA accuracy improvement: ~{p90_improvement:.1f}%")
    print(f"   - Better resource allocation: R² improved by {r2_improvement:.1f}%")
    
    return {
        'mae_improvement': mae_improvement,
        'rmse_improvement': rmse_improvement,
        'r2_improvement': r2_improvement,
        'p90_improvement': p90_improvement,
        'annual_savings': annual_savings
    }

def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print(" "*20 + "KPT PREDICTION MODEL TRAINING")
    print(" "*18 + "WITH 5 NEW FEATURES")
    print("="*80)
    
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Target variable
    y = data['true_kpt_minutes']
    
    # ============= BASELINE MODEL =============
    print("\n[1/2] Training Baseline Model...")
    X_baseline = create_baseline_features(data)
    
    # Train-test split
    X_train_base, X_test_base, y_train, y_test = train_test_split(
        X_baseline, y, test_size=0.2, random_state=42
    )
    
    baseline_model, base_mae, base_rmse, base_r2, base_p90 = train_baseline_model(
        X_train_base, y_train, X_test_base, y_test
    )
    
    # ============= ENHANCED MODEL =============
    print("\n[2/2] Training Enhanced Model with 5 NEW Features...")
    X_enhanced = create_enhanced_features(data)
    
    # Train-test split (using same split for fair comparison)
    X_train_enh, X_test_enh, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42
    )
    
    enhanced_model, enh_mae, enh_rmse, enh_r2, enh_p90, feat_imp = train_enhanced_model(
        X_train_enh, y_train, X_test_enh, y_test
    )
    
    # ============= COMPARISON =============
    improvements = compare_models(
        (base_mae, base_rmse, base_r2, base_p90),
        (enh_mae, enh_rmse, enh_r2, enh_p90)
    )
    
    # Save improvement metrics
    improvements_df = pd.DataFrame([improvements])
    improvements_df.to_csv('models/model_improvements.csv', index=False)
    
    # Save feature importance
    feat_imp.to_csv('models/feature_importance.csv', index=False)
    
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print("="*80)
    print(f"\nModels saved in 'models/' directory:")
    print(f"  - baseline_model.pkl")
    print(f"  - enhanced_model.pkl")
    print(f"  - model_improvements.csv")
    print(f"  - feature_importance.csv")
    
    print(f"\n⭐ New Features Summary:")
    print(f"  1. order_value_per_item - Complexity indicator")
    print(f"  2. dist_to_nearest_rush - Temporal dynamics")
    print(f"  3. kitchen_efficiency_score - Load vs capacity ratio")
    print(f"  4. order_complexity_score - Normalized item/value metric")
    print(f"  5. merchant_historical_bias - Historical error tracking")
    
    return baseline_model, enhanced_model, improvements

if __name__ == "__main__":
    baseline_model, enhanced_model, improvements = main()
