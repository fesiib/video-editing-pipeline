import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict

from analysis.analysis_constants import COLOR_MAPPING_EDITING_PROCESS

def timeline_plot(save_filename, events, start_times):
    # end times are just the next start times
    end_times = start_times[1:] + [start_times[-1]]

    # Convert time strings to datetime objects
    start_times = [datetime.strptime(time, "%H:%M:%S") for time in start_times]
    end_times = [datetime.strptime(time, "%H:%M:%S") for time in end_times]

    # Create a dictionary to group events by name and store unique events
    event_groups = defaultdict(list)
    unique_events = set(events)
    for event, start_time in zip(events, start_times):
        event_groups[event].append(start_time)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, len(unique_events)))

    # Plot the timeline events as colored points, aggregated by name
    unique_events = list(unique_events)
    unique_events.sort()
    # unique_events = ["ideate", "describe", "examine", "manual"]

    colors = []
    for event in unique_events:
        color = "black"
        for color_event in COLOR_MAPPING_EDITING_PROCESS.keys():
            if color_event in event:
                color = COLOR_MAPPING_EDITING_PROCESS[color_event]
                break
        colors.append(color)

    for i, event in enumerate(unique_events):
        event_times = event_groups[event]
        #event_positions = np.arange(len(event_times))
        event_positions = [i] * len(event_times)
        ax.scatter(event_times, event_positions, label=event, s=20, color=colors[i])

    # Customize the plot
    ax.set_yticks(np.arange(len(unique_events)))
    ax.set_yticklabels(unique_events)
    ax.set_xlabel("Timeline")
    ax.set_title("Aggregated Timeline Plot")
    ax.legend()

    # Display the plot
    # plt.show()

    # Save the plot
    fig.savefig(save_filename)
    plt.close(fig)

def pie_plot(save_filename, data, labels):
    label_data = {}
    for label, value in zip(labels, data):
        if label in label_data:
            label_data[label] += value
        else:
            label_data[label] = value
    # Extract aggregated labels and values
    aggregated_labels = list(label_data.keys())
    aggregated_labels.sort()
    aggregated_data = []
    # aggregated_labels = ["ideate", "describe", "examine", "manual"]
    for label in aggregated_labels:
        aggregated_data.append(label_data[label])

    # turn to percentage
    sum_data = np.sum(aggregated_data)
    aggregated_data = [value / sum_data * 100 for value in aggregated_data]

    print(aggregated_labels, aggregated_data)

    colors = []
    for event in aggregated_labels:
        color = "black"
        for color_event in COLOR_MAPPING_EDITING_PROCESS.keys():
            if color_event in event:
                color = COLOR_MAPPING_EDITING_PROCESS[color_event]
                break
        colors.append(color)

    # Create a pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        aggregated_data,
        labels=aggregated_labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
    )

    # Add a title
    ax.legend(aggregated_labels, loc="upper right")
    ax.set_title("Aggregated Pie Chart")

    # Display the plot
    # plt.show()

    # Save the plot
    fig.savefig(save_filename)

    plt.close(fig)

def bar_plot(save_filename, data, labels):
    label_data = {}
    for label, value in zip(labels, data):
        if label in label_data:
            label_data[label] += value
        else:
            label_data[label] = value

    # Extract aggregated labels and values
    aggregated_labels = list(label_data.keys())
    aggregated_labels.sort()
    aggregated_data = []
    # aggregated_labels = ["ideate", "describe", "examine", "manual"]
    for label in aggregated_labels:
        aggregated_data.append(label_data[label])

    # turn to percentage

    colors = []
    for event in aggregated_labels:
        color = "black"
        for color_event in COLOR_MAPPING_EDITING_PROCESS.keys():
            if color_event in event:
                color = COLOR_MAPPING_EDITING_PROCESS[color_event]
                break
        colors.append(color)
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(aggregated_labels, aggregated_data, color=colors)

    # Add a title
    # ax.legend(aggregated_labels, loc="upper right")
    ax.set_title("Aggregated Bar Chart")

    # Display the plot
    # plt.show()

    # Save the plot
    fig.savefig(save_filename)

    plt.close(fig)
