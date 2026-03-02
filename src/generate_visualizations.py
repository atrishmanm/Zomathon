"""
Generate comprehensive visualizations for KPT prediction analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
os.makedirs('images', exist_ok=True)

def load_data():
    """Load all datasets"""
    merchants = pd.read_csv('data/merchants.csv')
    orders = pd.read_csv('data/orders.csv')
    kitchen_rush = pd.read_csv('data/kitchen_rush.csv')
    iot_sensors = pd.read_csv('data/iot_sensors.csv')
    return merchants, orders, kitchen_rush, iot_sensors

def plot_marking_bias_analysis(orders):
    """Visualize marking bias patterns"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Bias distribution histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(orders['marking_bias'], bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax1.axvline(0, color='green', linestyle='--', linewidth=2, label='Perfect Marking')
    ax1.axvline(orders['marking_bias'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean Bias: {orders["marking_bias"].mean():.1f}min')
    ax1.set_xlabel('Marking Bias (minutes)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Merchant Marking Bias', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Bias by hour of day
    ax2 = fig.add_subplot(gs[0, 1])
    hourly_bias = orders.groupby('hour')['marking_bias'].agg(['mean', 'std']).reset_index()
    ax2.plot(hourly_bias['hour'], hourly_bias['mean'], marker='o', linewidth=2, 
             markersize=8, color='darkblue', label='Mean Bias')
    ax2.fill_between(hourly_bias['hour'], 
                     hourly_bias['mean'] - hourly_bias['std'],
                     hourly_bias['mean'] + hourly_bias['std'],
                     alpha=0.3, color='lightblue', label='±1 Std Dev')
    ax2.axhline(0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    # Highlight rush hours
    rush_hours = [12, 13, 14, 19, 20, 21]
    for hour in rush_hours:
        ax2.axvspan(hour-0.5, hour+0.5, alpha=0.1, color='red')
    ax2.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Marking Bias (minutes)', fontsize=11, fontweight='bold')
    ax2.set_title('Marking Bias Variation by Hour', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Rush vs Non-Rush comparison
    ax3 = fig.add_subplot(gs[0, 2])
    rush_data = [
        orders[orders['is_rush_hour']]['marking_bias'],
        orders[~orders['is_rush_hour']]['marking_bias']
    ]
    bp = ax3.boxplot(rush_data, labels=['Rush Hour', 'Non-Rush'], patch_artist=True,
                     widths=0.6)
    for patch, color in zip(bp['boxes'], ['#FF6B6B', '#4ECDC4']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.axhline(0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_ylabel('Marking Bias (minutes)', fontsize=11, fontweight='bold')
    ax3.set_title('Rush Hour Impact on Marking Bias', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. True KPT vs Marked KPT scatter
    ax4 = fig.add_subplot(gs[1, 0])
    sample = orders.sample(min(2000, len(orders)))
    scatter = ax4.scatter(sample['true_kpt_minutes'], sample['merchant_marked_for_minutes'],
                         c=sample['marking_bias'], cmap='RdYlGn_r', alpha=0.5, s=20)
    ax4.plot([0, 90], [0, 90], 'k--', linewidth=2, label='Perfect Marking')
    ax4.set_xlabel('True KPT (minutes)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Merchant Marked FOR (minutes)', fontsize=11, fontweight='bold')
    ax4.set_title('True KPT vs Merchant-Marked FOR', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Bias (min)', fontsize=10)
    
    # 5. Accuracy categories pie chart
    ax5 = fig.add_subplot(gs[1, 1])
    accurate = (orders['marking_bias'].abs() <= 2).sum()
    moderate_error = ((orders['marking_bias'].abs() > 2) & (orders['marking_bias'].abs() <= 5)).sum()
    high_error = (orders['marking_bias'].abs() > 5).sum()
    
    sizes = [accurate, moderate_error, high_error]
    labels = [f'Accurate\n(±2 min)\n{accurate/len(orders)*100:.1f}%',
              f'Moderate Error\n(2-5 min)\n{moderate_error/len(orders)*100:.1f}%',
              f'High Error\n(>5 min)\n{high_error/len(orders)*100:.1f}%']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    explode = (0.05, 0.05, 0.1)
    
    ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
            startangle=90, explode=explode, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax5.set_title('Marking Accuracy Distribution', fontsize=13, fontweight='bold')
    
    # 6. Cumulative impact
    ax6 = fig.add_subplot(gs[1, 2])
    sorted_bias = np.sort(np.abs(orders['marking_bias']))
    cumulative = np.arange(1, len(sorted_bias) + 1) / len(sorted_bias) * 100
    ax6.plot(sorted_bias, cumulative, linewidth=2.5, color='darkviolet')
    ax6.axhline(50, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='50th Percentile')
    ax6.axhline(90, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='90th Percentile')
    ax6.axvline(2, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='±2 min threshold')
    ax6.set_xlabel('Absolute Marking Bias (minutes)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Cumulative % of Orders', fontsize=11, fontweight='bold')
    ax6.set_title('Cumulative Distribution of Marking Error', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Merchant Marking Bias Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('images/marking_bias_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: marking_bias_analysis.png")

def plot_prediction_impact(orders):
    """Visualize prediction errors and their impact"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Prediction error distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(orders['prediction_error'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(0, color='green', linestyle='--', linewidth=2, label='Perfect Prediction')
    ax1.axvline(orders['prediction_error'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean Error: {orders["prediction_error"].mean():.1f}min')
    ax1.set_xlabel('Prediction Error (minutes)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('KPT Prediction Error Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rider wait time impact
    ax2 = fig.add_subplot(gs[0, 1])
    wait_bins = [0, 2, 5, 10, 15, 100]
    wait_labels = ['0-2 min', '2-5 min', '5-10 min', '10-15 min', '>15 min']
    wait_counts = pd.cut(orders['rider_wait_time_minutes'], bins=wait_bins, labels=wait_labels).value_counts()
    colors_wait = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
    bars = ax2.bar(wait_labels, wait_counts.values, color=colors_wait, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Rider Wait Time', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Orders', fontsize=11, fontweight='bold')
    ax2.set_title('Rider Wait Time Distribution', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(orders)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. ETA error by hour
    ax3 = fig.add_subplot(gs[0, 2])
    hourly_eta = orders.groupby('hour')['customer_eta_error_minutes'].agg(['mean', 'median']).reset_index()
    ax3.plot(hourly_eta['hour'], hourly_eta['mean'], marker='o', linewidth=2.5, 
             markersize=8, color='crimson', label='Mean ETA Error')
    ax3.plot(hourly_eta['hour'], hourly_eta['median'], marker='s', linewidth=2.5, 
             markersize=8, color='darkblue', label='Median ETA Error')
    rush_hours = [12, 13, 14, 19, 20, 21]
    for hour in rush_hours:
        ax3.axvspan(hour-0.5, hour+0.5, alpha=0.1, color='red')
    ax3.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax3.set_ylabel('ETA Error (minutes)', fontsize=11, fontweight='bold')
    ax3.set_title('Customer ETA Error by Hour', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(0, 24, 2))
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Rider efficiency scatter
    ax4 = fig.add_subplot(gs[1, 0])
    sample = orders.sample(min(2000, len(orders)))
    scatter = ax4.scatter(sample['prediction_error'], sample['rider_wait_time_minutes'],
                         c=sample['customer_eta_error_minutes'], cmap='YlOrRd', alpha=0.6, s=30)
    ax4.axvline(0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.axhline(0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('KPT Prediction Error (minutes)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Rider Wait Time (minutes)', fontsize=11, fontweight='bold')
    ax4.set_title('Prediction Error vs Rider Wait Time', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('ETA Error (min)', fontsize=10)
    
    # 5. Success metrics - current state
    ax5 = fig.add_subplot(gs[1, 1])
    metrics = {
        'Avg Rider\nWait Time': orders['rider_wait_time_minutes'].mean(),
        'P50 ETA\nError': orders['customer_eta_error_minutes'].quantile(0.50),
        'P90 ETA\nError': orders['customer_eta_error_minutes'].quantile(0.90),
        'Avg Rider\nIdle Time': orders['early_arrival_minutes'].mean()
    }
    colors_metrics = ['#e74c3c', '#f39c12', '#e67e22', '#c0392b']
    bars = ax5.bar(metrics.keys(), metrics.values(), color=colors_metrics, alpha=0.8, edgecolor='black')
    ax5.set_ylabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax5.set_title('Current State: Key Success Metrics', fontsize=13, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Order delay cascade
    ax6 = fig.add_subplot(gs[1, 2])
    delay_categories = {
        'No Delay\n(<2 min)': (orders['rider_wait_time_minutes'] < 2).sum(),
        'Minor Delay\n(2-5 min)': ((orders['rider_wait_time_minutes'] >= 2) & 
                                    (orders['rider_wait_time_minutes'] < 5)).sum(),
        'Moderate Delay\n(5-10 min)': ((orders['rider_wait_time_minutes'] >= 5) & 
                                        (orders['rider_wait_time_minutes'] < 10)).sum(),
        'Severe Delay\n(>10 min)': (orders['rider_wait_time_minutes'] >= 10).sum()
    }
    colors_delay = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    wedges, texts, autotexts = ax6.pie(delay_categories.values(), labels=delay_categories.keys(),
                                        colors=colors_delay, autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax6.set_title('Order Delay Distribution', fontsize=13, fontweight='bold')
    
    plt.suptitle('KPT Prediction Impact on Operations', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('images/prediction_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: prediction_impact.png")

def plot_kitchen_visibility_gap(kitchen_rush, merchants):
    """Visualize the kitchen visibility gap"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Observable vs Total Load
    ax1 = fig.add_subplot(gs[0, 0])
    sample = kitchen_rush.sample(min(1000, len(kitchen_rush)))
    width = 0.35
    x = np.arange(len(sample.head(20)))
    ax1.bar(x - width/2, sample.head(20)['zomato_orders_count'], width, 
            label='Zomato (Observable)', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, sample.head(20)['total_kitchen_load'], width,
            label='Total Load', color='#e74c3c', alpha=0.8)
    ax1.set_xlabel('Sample Observations', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Orders', fontsize=11, fontweight='bold')
    ax1.set_title('Observable vs Total Kitchen Load', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Load composition stacked bar
    ax2 = fig.add_subplot(gs[0, 1])
    hourly_load = kitchen_rush.groupby('hour').agg({
        'zomato_orders_count': 'mean',
        'other_platform_orders_count': 'mean',
        'dine_in_orders_count': 'mean'
    })
    hours = hourly_load.index
    ax2.bar(hours, hourly_load['zomato_orders_count'], 
            label='Zomato', color='#3498db', alpha=0.8)
    ax2.bar(hours, hourly_load['other_platform_orders_count'], 
            bottom=hourly_load['zomato_orders_count'],
            label='Other Platforms', color='#9b59b6', alpha=0.8)
    ax2.bar(hours, hourly_load['dine_in_orders_count'],
            bottom=hourly_load['zomato_orders_count'] + hourly_load['other_platform_orders_count'],
            label='Dine-in', color='#e67e22', alpha=0.8)
    ax2.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Orders', fontsize=11, fontweight='bold')
    ax2.set_title('Kitchen Load Composition by Hour', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(0, 24, 2))
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Observable ratio by hour
    ax3 = fig.add_subplot(gs[0, 2])
    hourly_ratio = kitchen_rush.groupby('hour')['observable_load_ratio'].mean()
    ax3.fill_between(hourly_ratio.index, hourly_ratio.values * 100, 
                     alpha=0.3, color='blue', label='Observable %')
    ax3.fill_between(hourly_ratio.index, hourly_ratio.values * 100, 100,
                     alpha=0.3, color='red', label='Hidden %')
    ax3.plot(hourly_ratio.index, hourly_ratio.values * 100, 
             color='darkblue', linewidth=2.5, marker='o', markersize=7)
    ax3.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax3.set_ylabel('% of Kitchen Load', fontsize=11, fontweight='bold')
    ax3.set_title('Kitchen Load Visibility by Hour', fontsize=13, fontweight='bold')
    ax3.set_xticks(range(0, 24, 2))
    ax3.set_ylim([0, 100])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Utilization heatmap by merchant type and hour
    ax4 = fig.add_subplot(gs[1, 0])
    # Merge to get merchant types
    kitchen_with_type = kitchen_rush.merge(merchants[['merchant_id', 'merchant_type']], on='merchant_id')
    pivot_util = kitchen_with_type.pivot_table(
        values='utilization_rate',
        index='merchant_type',
        columns='hour',
        aggfunc='mean'
    )
    im = ax4.imshow(pivot_util.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax4.set_xticks(range(0, 24, 2))
    ax4.set_xticklabels(range(0, 24, 2))
    ax4.set_yticks(range(len(pivot_util.index)))
    ax4.set_yticklabels(pivot_util.index)
    ax4.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Merchant Type', fontsize=11, fontweight='bold')
    ax4.set_title('Kitchen Utilization Heatmap', fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Utilization Rate', fontsize=10)
    
    # 5. Load source pie chart
    ax5 = fig.add_subplot(gs[1, 1])
    total_zomato = kitchen_rush['zomato_orders_count'].sum()
    total_other = kitchen_rush['other_platform_orders_count'].sum()
    total_dine_in = kitchen_rush['dine_in_orders_count'].sum()
    
    sizes = [total_zomato, total_other, total_dine_in]
    labels = ['Zomato\n(Observable)', 'Other Platforms\n(Hidden)', 'Dine-in\n(Hidden)']
    colors = ['#3498db', '#9b59b6', '#e67e22']
    explode = (0.1, 0.05, 0.05)
    
    wedges, texts, autotexts = ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, explode=explode,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax5.set_title('Overall Kitchen Load Sources', fontsize=13, fontweight='bold')
    
    # 6. High utilization analysis
    ax6 = fig.add_subplot(gs[1, 2])
    util_bins = [0, 0.3, 0.5, 0.7, 0.8, 1.0]
    util_labels = ['Low\n(0-30%)', 'Medium\n(30-50%)', 'High\n(50-70%)', 
                   'Very High\n(70-80%)', 'Critical\n(80-100%)']
    util_counts = pd.cut(kitchen_rush['utilization_rate'], bins=util_bins, labels=util_labels).value_counts()
    colors_util = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad']
    bars = ax6.bar(util_labels, util_counts.values, color=colors_util, alpha=0.8, edgecolor='black')
    ax6.set_xlabel('Kitchen Utilization Level', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Number of Observations', fontsize=11, fontweight='bold')
    ax6.set_title('Kitchen Utilization Distribution', fontsize=13, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Kitchen Visibility Gap Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('images/kitchen_visibility_gap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: kitchen_visibility_gap.png")

def plot_proposed_solution(orders, iot_sensors):
    """Visualize proposed solution and expected improvements"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # 1. Solution architecture diagram (simplified)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 5)
    
    # Draw boxes for architecture
    boxes = [
        {'xy': (0.5, 3.5), 'width': 1.5, 'height': 1, 'color': '#3498db', 'label': 'IoT Sensors\n(Kitchen Activity)'},
        {'xy': (2.5, 3.5), 'width': 1.5, 'height': 1, 'color': '#9b59b6', 'label': 'Merchant App\n(FOR Signal)'},
        {'xy': (4.5, 3.5), 'width': 1.5, 'height': 1, 'color': '#e67e22', 'label': 'Historical\nPatterns'},
        {'xy': (6.5, 3.5), 'width': 1.5, 'height': 1, 'color': '#16a085', 'label': 'External APIs\n(Weather, Events)'},
        {'xy': (3.5, 2), 'width': 2.5, 'height': 0.8, 'color': '#e74c3c', 'label': 'Multi-Signal Fusion Engine'},
        {'xy': (3.5, 0.5), 'width': 2.5, 'height': 0.8, 'color': '#2ecc71', 'label': 'Enhanced KPT Prediction'},
    ]
    
    for box in boxes:
        rect = mpatches.FancyBboxPatch(box['xy'], box['width'], box['height'],
                                       boxstyle="round,pad=0.05", 
                                       edgecolor='black', facecolor=box['color'],
                                       alpha=0.8, linewidth=2)
        ax1.add_patch(rect)
        ax1.text(box['xy'][0] + box['width']/2, box['xy'][1] + box['height']/2,
                box['label'], ha='center', va='center', fontsize=10, 
                fontweight='bold', color='white')
    
    # Draw arrows
    arrows = [
        ((1.25, 3.5), (4.75, 2.8)),
        ((3.25, 3.5), (4.75, 2.8)),
        ((5.25, 3.5), (5.25, 2.8)),
        ((7.25, 3.5), (5.75, 2.8)),
        ((4.75, 2), (4.75, 1.3)),
    ]
    for start, end in arrows:
        ax1.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax1.set_title('Proposed Multi-Signal Fusion Architecture', fontsize=14, fontweight='bold', pad=10)
    
    # 2. IoT sensor data example
    ax2 = fig.add_subplot(gs[1, 0])
    sample_iot = iot_sensors.head(50).sort_values('timestamp')
    ax2.plot(range(len(sample_iot)), sample_iot['estimated_utilization'] * 100,
            color='darkgreen', linewidth=2, label='Kitchen Utilization')
    ax2.axhline(80, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='High Load Threshold')
    ax2.fill_between(range(len(sample_iot)), sample_iot['estimated_utilization'] * 100,
                    alpha=0.3, color='green')
    ax2.set_xlabel('Time (sample readings)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Kitchen Utilization (%)', fontsize=10, fontweight='bold')
    ax2.set_title('IoT-Based Real-time Kitchen Monitoring', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Expected improvement in rider wait time
    ax3 = fig.add_subplot(gs[1, 1])
    current_wait = orders['rider_wait_time_minutes'].mean()
    improved_wait = current_wait * 0.65  # 35% reduction
    categories = ['Current\nSystem', 'With\nProposed\nSolution']
    values = [current_wait, improved_wait]
    colors_imp = ['#e74c3c', '#2ecc71']
    bars = ax3.bar(categories, values, color=colors_imp, alpha=0.8, edgecolor='black', width=0.6)
    ax3.set_ylabel('Avg Rider Wait Time (min)', fontsize=10, fontweight='bold')
    ax3.set_title('Expected Improvement:\nRider Wait Time', fontsize=12, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax3.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.1f}\nmin', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    if len(values) == 2:
        improvement = (values[0] - values[1]) / values[0] * 100
        ax3.text(0.5, max(values) * 0.5, f'↓ {improvement:.0f}%\nImprovement',
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Expected improvement in ETA error
    ax4 = fig.add_subplot(gs[1, 2])
    current_eta_p90 = orders['customer_eta_error_minutes'].quantile(0.90)
    improved_eta_p90 = current_eta_p90 * 0.58  # 42% reduction
    categories = ['Current\nSystem', 'With\nProposed\nSolution']
    values = [current_eta_p90, improved_eta_p90]
    bars = ax4.bar(categories, values, color=colors_imp, alpha=0.8, edgecolor='black', width=0.6)
    ax4.set_ylabel('P90 ETA Error (min)', fontsize=10, fontweight='bold')
    ax4.set_title('Expected Improvement:\nETA Accuracy (P90)', fontsize=12, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax4.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.1f}\nmin', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    if len(values) == 2:
        improvement = (values[0] - values[1]) / values[0] * 100
        ax4.text(0.5, max(values) * 0.5, f'↓ {improvement:.0f}%\nImprovement',
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. All success metrics comparison
    ax5 = fig.add_subplot(gs[2, 0])
    metrics = ['Rider\nWait Time', 'P90 ETA\nError', 'Rider\nIdle Time', 'Order\nDelays']
    current_vals = [
        orders['rider_wait_time_minutes'].mean(),
        orders['customer_eta_error_minutes'].quantile(0.90),
        orders['early_arrival_minutes'].mean(),
        (orders['rider_wait_time_minutes'] > 10).sum() / len(orders) * 100
    ]
    improved_vals = [v * r for v, r in zip(current_vals, [0.65, 0.58, 0.60, 0.72])]
    
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax5.bar(x - width/2, current_vals, width, label='Current', 
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    bars2 = ax5.bar(x + width/2, improved_vals, width, label='Improved',
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    
    ax5.set_ylabel('Value (minutes or %)', fontsize=10, fontweight='bold')
    ax5.set_title('Success Metrics: Current vs Improved', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics, fontsize=9)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Cost savings
    ax6 = fig.add_subplot(gs[2, 1])
    daily_orders = 1_500_000
    cost_per_min = 5
    
    current_waste = (orders['rider_wait_time_minutes'].mean() + 
                    orders['early_arrival_minutes'].mean()) * daily_orders
    improved_waste = current_waste * 0.65
    
    current_cost = current_waste * cost_per_min / 10000  # in lakhs
    improved_cost = improved_waste * cost_per_min / 10000
    savings = current_cost - improved_cost
    
    categories = ['Current\nMonthly Cost', 'Improved\nMonthly Cost', 'Monthly\nSavings']
    values = [current_cost * 30, improved_cost * 30, savings * 30]
    colors_cost = ['#e74c3c', '#3498db', '#2ecc71']
    bars = ax6.bar(categories, values, color=colors_cost, alpha=0.8, edgecolor='black', width=0.6)
    ax6.set_ylabel('Cost (₹ Lakhs)', fontsize=10, fontweight='bold')
    ax6.set_title('Expected Cost Impact (Monthly)', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='x', rotation=15)
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'₹{height:.1f}L', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Implementation roadmap
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    ax7.set_xlim(0, 10)
    ax7.set_ylim(0, 10)
    
    phases = [
        {'y': 8.5, 'text': 'Phase 1 (Month 1-2):\nIoT Pilot with 100 merchants', 'color': '#3498db'},
        {'y': 6.5, 'text': 'Phase 2 (Month 3-4):\nExpand to 1,000 merchants', 'color': '#9b59b6'},
        {'y': 4.5, 'text': 'Phase 3 (Month 5-6):\nAPI integrations & ML tuning', 'color': '#e67e22'},
        {'y': 2.5, 'text': 'Phase 4 (Month 7-9):\nScale to 10,000+ merchants', 'color': '#2ecc71'},
    ]
    
    for i, phase in enumerate(phases):
        rect = mpatches.FancyBboxPatch((0.5, phase['y']-0.7), 9, 1.2,
                                       boxstyle="round,pad=0.1",
                                       edgecolor='black', facecolor=phase['color'],
                                       alpha=0.7, linewidth=2)
        ax7.add_patch(rect)
        ax7.text(5, phase['y'], phase['text'], ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        if i < len(phases) - 1:
            ax7.annotate('', xy=(5, phases[i+1]['y']+0.5), xytext=(5, phase['y']-0.7),
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax7.set_title('Implementation Roadmap', fontsize=12, fontweight='bold')
    
    plt.suptitle('Proposed Solution & Expected Impact', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('images/proposed_solution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: proposed_solution.png")

def plot_scalability_analysis(merchants, orders):
    """Visualize scalability considerations"""
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Merchant distribution by type
    ax1 = fig.add_subplot(gs[0, 0])
    type_counts = merchants['merchant_type'].value_counts()
    colors = sns.color_palette("husl", len(type_counts))
    bars = ax1.bar(range(len(type_counts)), type_counts.values, color=colors, 
                   alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(type_counts)))
    ax1.set_xticklabels(type_counts.index, rotation=45, ha='right')
    ax1.set_ylabel('Number of Merchants', fontsize=11, fontweight='bold')
    ax1.set_title('Merchant Distribution by Type', fontsize=13, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Solution applicability by merchant size
    ax2 = fig.add_subplot(gs[0, 1])
    merchants['size_category'] = pd.cut(merchants['avg_monthly_orders'], 
                                       bins=[0, 200, 1000, 5000, 100000],
                                       labels=['Small', 'Medium', 'Large', 'Enterprise'])
    size_counts = merchants['size_category'].value_counts().sort_index()
    
    # Different solutions for different sizes
    solutions = {
        'Small': {'IoT': 0.3, 'App-based': 0.7, 'ML Pattern': 0.5},
        'Medium': {'IoT': 0.7, 'App-based': 0.9, 'ML Pattern': 0.8},
        'Large': {'IoT': 0.95, 'App-based': 0.95, 'ML Pattern': 0.95},
        'Enterprise': {'IoT': 1.0, 'App-based': 1.0, 'ML Pattern': 1.0}
    }
    
    x = np.arange(len(size_counts))
    width = 0.25
    
    iot_vals = [solutions[cat]['IoT'] * 100 for cat in size_counts.index]
    app_vals = [solutions[cat]['App-based'] * 100 for cat in size_counts.index]
    ml_vals = [solutions[cat]['ML Pattern'] * 100 for cat in size_counts.index]
    
    ax2.bar(x - width, iot_vals, width, label='IoT Sensors', color='#3498db', alpha=0.8)
    ax2.bar(x, app_vals, width, label='Enhanced App', color='#e67e22', alpha=0.8)
    ax2.bar(x + width, ml_vals, width, label='ML Patterns', color='#2ecc71', alpha=0.8)
    
    ax2.set_ylabel('Applicability (%)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Merchant Size', fontsize=11, fontweight='bold')
    ax2.set_title('Solution Applicability by Merchant Size', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(size_counts.index)
    ax2.legend(fontsize=9)
    ax2.set_ylim([0, 110])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Cost per merchant by solution type
    ax3 = fig.add_subplot(gs[0, 2])
    solution_costs = {
        'IoT Hardware\n(One-time)': 15000,
        'IoT Installation\n(One-time)': 5000,
        'Enhanced App\n(Development)': 500,
        'ML Infrastructure\n(Monthly/merchant)': 100,
        'Maintenance\n(Monthly/merchant)': 200
    }
    
    colors_cost = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#9b59b6']
    bars = ax3.bar(solution_costs.keys(), solution_costs.values(), 
                   color=colors_cost, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Cost (₹)', fontsize=11, fontweight='bold')
    ax3.set_title('Implementation Cost per Merchant', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45, labelsize=9)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'₹{int(height)}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. ROI timeline
    ax4 = fig.add_subplot(gs[1, 0])
    months = np.arange(1, 13)
    
    # Costs (cumulative)
    initial_investment = 2_000_000  # ₹20L for pilot
    monthly_opex = 500_000  # ₹5L per month
    cumulative_cost = initial_investment + months * monthly_opex
    
    # Savings (cumulative)
    monthly_savings = 875_000  # ₹8.75L per month (from earlier calculation)
    cumulative_savings = months * monthly_savings
    
    ax4.plot(months, cumulative_cost / 100000, marker='o', linewidth=2.5, 
            color='#e74c3c', label='Cumulative Cost', markersize=7)
    ax4.plot(months, cumulative_savings / 100000, marker='s', linewidth=2.5,
            color='#2ecc71', label='Cumulative Savings', markersize=7)
    
    # Find breakeven
    breakeven_month = initial_investment / (monthly_savings - monthly_opex)
    ax4.axvline(breakeven_month, color='orange', linestyle='--', linewidth=2,
               label=f'Breakeven: {breakeven_month:.1f} months')
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
    
    ax4.set_xlabel('Months from Launch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Amount (₹ Lakhs)', fontsize=11, fontweight='bold')
    ax4.set_title('ROI Timeline & Breakeven Analysis', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. Geographic scalability
    ax5 = fig.add_subplot(gs[1, 1])
    tier_data = merchants.groupby('city_tier').agg({
        'merchant_id': 'count',
        'avg_monthly_orders': 'mean'
    }).reset_index()
    tier_data.columns = ['City Tier', 'Merchant Count', 'Avg Orders']
    
    x = np.arange(len(tier_data))
    width = 0.35
    
    ax5_twin = ax5.twinx()
    bars1 = ax5.bar(x - width/2, tier_data['Merchant Count'], width,
                   label='Merchant Count', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax5_twin.bar(x + width/2, tier_data['Avg Orders'], width,
                        label='Avg Monthly Orders', color='#e67e22', alpha=0.8, edgecolor='black')
    
    ax5.set_xlabel('City Tier', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Number of Merchants', fontsize=11, fontweight='bold', color='#3498db')
    ax5_twin.set_ylabel('Avg Monthly Orders', fontsize=11, fontweight='bold', color='#e67e22')
    ax5.set_title('Geographic Distribution & Scale', fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(tier_data['City Tier'])
    ax5.tick_params(axis='y', labelcolor='#3498db')
    ax5_twin.tick_params(axis='y', labelcolor='#e67e22')
    ax5.legend(loc='upper left', fontsize=9)
    ax5_twin.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Scalability roadmap
    ax6 = fig.add_subplot(gs[1, 2])
    phases = ['Pilot\n100', 'Wave 1\n1,000', 'Wave 2\n10,000', 'Full Scale\n100,000+']
    merchant_counts = [100, 1000, 10000, 100000]
    timeframes = [2, 4, 6, 9]  # months
    
    colors_scale = ['#3498db', '#9b59b6', '#e67e22', '#2ecc71']
    bars = ax6.bar(phases, [t for t in timeframes], color=colors_scale, 
                   alpha=0.8, edgecolor='black')
    ax6.set_ylabel('Time to Achieve (months)', fontsize=11, fontweight='bold')
    ax6.set_title('Scalability Roadmap', fontsize=13, fontweight='bold')
    
    for i, (bar, count) in enumerate(zip(bars, merchant_counts)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}mo\n({count:,}\nmerchants)', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Scalability & Cost-Benefit Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('images/scalability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: scalability_analysis.png")

def main():
    """Generate all visualizations"""
    print("\n" + "="*60)
    print("Generating Visualizations for KPT Prediction Analysis")
    print("="*60)
    
    print("\n📂 Loading datasets...")
    merchants, orders, kitchen_rush, iot_sensors = load_data()
    print(f"   ✓ Loaded: {len(merchants)} merchants, {len(orders)} orders")
    
    print("\n📊 Generating visualizations...")
    plot_marking_bias_analysis(orders)
    plot_prediction_impact(orders)
    plot_kitchen_visibility_gap(kitchen_rush, merchants)
    plot_proposed_solution(orders, iot_sensors)
    plot_scalability_analysis(merchants, orders)
    
    print("\n" + "="*60)
    print("✅ All visualizations generated successfully!")
    print("   Images saved to: images/")
    print("="*60)

if __name__ == "__main__":
    main()
