from bs4 import BeautifulSoup as WebScraper
from entry_datetime import entry_date, entry_datetime
from urllib.request import Request, urlopen

import pandas as pd
import numpy as np

!pip install pandas-profiling[notebook]
!pip install --upgrade numba
!pip install --upgrade pandas-profiling
!pip install --upgrade visions
!pip uninstall -y pandas-profiling
!pip install pandas-profiling==3.6.6 pydantic==1.10.12
!pip install ydata-profiling

import matplotlib.pyplot as visual_plt
import plotly.graph_objects as graph_obj
import plotly.express as px_graph
import plotly.offline as py_off
import seaborn as sns
import gc
import warnings
warnings.filterwarnings("ignore")

from ydata_profiling import Procsv_fileReport

current_day = entry_datetime.now()
prior_day_str = "%s %d,%d" % (entry_date.today().strftime("%b"), current_day.day-1, current_day.year)
prior_day_str

target_url = "https://www.worldometers.info/coronavirus/#countries"
request_obj = Request(target_url, headers={'User-Agent': 'Mozilla/5.0'})

web_response = urlopen(request_obj)
parsed_html = WebScraper(web_response, "html.parser")
parsed_html

data_table = parsed_html.findAll("table", {"id": "main_table_countries_yesterday"})
row_elements = data_table[0].findAll("tr", {"style": ""})
header_row = row_elements[0]

del row_elements[0]

extracted_data = []
process_clean = True

for nation in row_elements:
    nation_data = []
    nation_columns = nation.findAll("td")

    if nation_columns[1].text == "China":
        continue

    for i in range(1, len(nation_columns)):
        processed_value = nation_columns[i].text
        if process_clean:
            if i != 1 and i != len(nation_columns)-1:
                processed_value = processed_value.replace(",", "")

                if processed_value.find('+') != -1:
                    processed_value = processed_value.replace("+", "")
                    processed_value = float(processed_value)

                elif processed_value.find("-") != -1:
                    processed_value = processed_value.replace("-", "")
                    processed_value = float(processed_value) * -1

        if processed_value == 'N/A':
            processed_value = 0
        elif processed_value == "" or processed_value == " ":
            processed_value = -1

        nation_data.append(processed_value)

    extracted_data.append(nation_data)

extracted_data

import pandas as pd

virus_dataset = pd.DataFrame(extracted_data)
virus_dataset.drop([15, 16, 17], inplace=True, axis=1)
virus_dataset.head()

field_names = ["Country", "Total Cases", "New Cases", "Total Deaths", "New Deaths",
              "Total Recovered", "New Recovered", "Active Cases", "Serious/Critical",
              "Total Cases/1M", "Deaths/1M", "Total Tests", "Test/1M", "Population", "Continent",
              "Column16", "Column17", "Column18"]

virus_dataset.columns = field_names

virus_dataset.head()

for field in virus_dataset.columns:
    if field != "Country" and field != "Continent":
        virus_dataset[field] = pd.to_numeric(virus_dataset[field], errors='coerce')
        virus_dataset[field] = virus_dataset[field].fillna(0)

virus_dataset["%Inc Cases"] = virus_dataset["New Cases"] / virus_dataset["Total Cases"] * 100
virus_dataset["%Inc Deaths"] = virus_dataset["New Deaths"] / virus_dataset["Total Deaths"] * 100
virus_dataset["%Inc Recovered"] = virus_dataset["New Recovered"] / virus_dataset["Total Recovered"] * 100

virus_dataset.head()

infection_summary = virus_dataset[["Total Recovered", "Active Cases", "Total Deaths"]].loc[0]

infection_breakdown = pd.DataFrame(infection_summary).reset_index()
infection_breakdown.columns = ["Category", "Count"]

infection_breakdown["Percentage"] = np.round(100 * infection_breakdown["Count"] / np.sum(infection_breakdown["Count"]), 2)
infection_breakdown["Disease"] = ["COVID-19" for i in range(len(infection_breakdown))]

visual = px_graph.bar(infection_breakdown, x="Disease", y="Percentage", color="Category", hover_data=["Count"])
visual.show()

daily_changes = virus_dataset[["New Cases", "New Recovered", "New Deaths"]].loc[0]

daily_summary = pd.DataFrame(daily_changes).reset_index()
daily_summary.columns = ["Category", "Count"]

daily_summary["Percentage"] = np.round(100 * daily_summary["Count"] / np.sum(daily_summary["Count"]), 2)
daily_summary["Disease"] = ["COVID-19" for i in range(len(daily_summary))]

visual = px_graph.bar(daily_summary, x="Disease", y="Percentage", color="Category", hover_data=["Count"])
visual.show()

growth_indicators = np.round(virus_dataset[["%Inc Cases", "%Inc Deaths", "%Inc Recovered"]].loc[0], 2)

metrics_display = pd.DataFrame(growth_indicators)
metrics_display.columns = ["Percentage"]

visual = graph_obj.Figure()

visual.add_trace(graph_obj.Bar(x=metrics_display.index, y=metrics_display["Percentage"], marker_color=["Yellow", "blue", "red"]))
visual.show()

regional_data = virus_dataset.groupby("Continent").sum()

if "All" in regional_data.index:
    regional_data = regional_data.drop("All")

regional_data = regional_data.reset_index()
regional_data

def display_regional_analysis(metric_list):
    for metric in metric_list:
        region_subset = regional_data[["Continent", metric]]
        region_subset["Percentage"] = np.round(100 * region_subset[metric] / np.sum(region_subset[metric]), 2)
        region_subset["Disease"] = ["COVID-19" for _ in range(len(region_subset))]

        visual = px_graph.bar(
            region_subset,
            x="Disease",
            y="Percentage",
            color="Continent",
            hover_data=[metric],
            text="Percentage"
        )

        visual.upentry_date_layout(
            title={"text": f"{metric}", "x": 0.5, "xanchor": "center"},
            yaxis_title="Percentage (%)",
            xaxis_title="Disease",
            font=dict(size=14),
            plot_bgcolor="rgba(0,0,0,0)"
        )

        visual.upentry_date_traces(textposition='inside')
        visual.show()
        gc.collect()

case_metrics = ["Total Cases", "Active Cases", "New Cases", "Serious/Critical", "Total Cases/1M"]
mortality_metrics = ["Total Deaths", "New Deaths", "Deaths/1M"]
recovery_metrics = ["Total Recovered", "New Recovered", "%Inc Recovered"]

display_regional_analysis(case_metrics)

virus_dataset = virus_dataset.drop([len(virus_dataset) - 1])
national_data = virus_dataset.drop([0])

national_data

DISPLAY_COUNT = 5
metric_columns = national_data.columns[1:14]

visual = graph_obj.Figure()
counter = 0
for i in national_data.index:
    if counter < DISPLAY_COUNT:
        visual.add_trace(graph_obj.Bar(
            name = national_data['Country'][i],
            x = metric_columns,
            y = national_data.loc[i][1:14]
        ))
    else:
        break
    counter += 1

visual.upentry_date_layout(
    title = {"text": f"top {DISPLAY_COUNT} countries affected"},
    yaxis_type = "log"
)
visual.show()

import os
import pandas as pd
import matplotlib.pyplot as visual_plt
from statsarima_models.tsa.arima.arima_model import ARIMA
import matplotlib.entry_dates as mentry_dates

data_directory = "data"
csv_file_list = sorted([f for f in os.listdir(data_directory) if f.endswith(".csv")])
virus_files = {}

for file_name in csv_file_list:
    try:
        loaded_data = pd.read_csv(os.path.join(data_directory, file_name))
        virus_files[file_name] = loaded_data
    except Exception as e:
        print(f"[ERROR] {file_name}: {e}")

focus_regions = ["California", "Texas", "New York"]
temporal_data = []

for file_name, dataset in virus_files.items():
    try:
        record_date = pd.to_entry_datetime(file_name.replace(".csv", ""), format="%m-%d-%Y")
    except:
        continue

    if "Province_State" not in dataset.columns:
        continue

    for region in focus_regions:
        region_data = dataset[dataset["Province_State"] == region]
        if not region_data.empty:
            temporal_data.append({
                "Date": record_date,
                "State": region,
                "Confirmed": region_data["Confirmed"].values[0] if "Confirmed" in region_data else None,
                "Deaths": region_data["Deaths"].values[0] if "Deaths" in region_data else None
            })

timeline_data = pd.DataFrame(temporal_data)
timeline_data["Date"] = pd.to_entry_datetime(timeline_data["Date"])
timeline_data = timeline_data.sort_values("Date")

visual_plt.charture(chartsize=(10, 6))
for region in focus_regions:
    region_timeline = timeline_data[timeline_data["State"] == region]
    visual_plt.plot(region_timeline["Date"], region_timeline["Confirmed"], label=f"{region} - Confirmed", linewidth=2)
    visual_plt.plot(region_timeline["Date"], region_timeline["Deaths"], linestyle="--", label=f"{region} - Deaths", linewidth=1.5)
visual_plt.title("COVID-19 Line Trend (2020–2023): Confirmed & Deaths")
visual_plt.xlabel("Date")
visual_plt.ylabel("Cases")
visual_plt.xticks(rotation=45)
visual_plt.grid(True)
visual_plt.legend()
visual_plt.tight_layout()
visual_plt.savechart("line_trend.png")
visual_plt.clf()

current_dataset = virus_files[csv_file_list[-1]]
if "Province_State" in current_dataset.columns and "Confirmed" in current_dataset.columns:
    current_dataset = current_dataset[current_dataset["Confirmed"] > 0]
    leading_regions = current_dataset.sort_values("Confirmed", ascending=False).head(10)
    visual_plt.bar(leading_regions["Province_State"].astype(str), leading_regions["Confirmed"].astype(int))
    visual_plt.title("Top 10 States by Confirmed Cases")
    visual_plt.xticks(rotation=45)
    visual_plt.tight_layout()
    visual_plt.savechart("bar_chart.png")
    visual_plt.clf()

if "Deaths" in current_dataset.columns:
    visual_plt.pie(leading_regions["Deaths"].fillna(0), labels=leading_regions["Province_State"], autopct='%1.1f%%')
    visual_plt.title("Death Distribution in Top 10 States")
    visual_plt.tight_layout()
    visual_plt.savechart("pie_chart.png")
    visual_plt.clf()

if "Confirmed" in current_dataset.columns and "Deaths" in current_dataset.columns:
    visual_plt.scatter(current_dataset["Confirmed"], current_dataset["Deaths"], alpha=0.5)
    visual_plt.xscale("log")
    visual_plt.yscale("log")
    visual_plt.title("Confirmed vs Deaths (Log Scale)")
    visual_plt.xlabel("Confirmed")
    visual_plt.ylabel("Deaths")
    visual_plt.grid(True)
    visual_plt.tight_layout()
    visual_plt.savechart("scatter_plot.png")
    visual_plt.clf()

california_dataset = timeline_data[timeline_data["State"] == "California"].sort_values("Date")
california_dataset.set_index("Date", inplace=True)
recent_ca_data = california_dataset["Confirmed"].last("180D")

if not recent_ca_data.empty:
    try:
        forecast_model = ARIMA(recent_ca_data, order=(2, 1, 2))
        fitted_model = forecast_model.fit()
        forecast_values = fitted_model.predicted_trend(steps=30)
        forecast_dates = pd.entry_date_range(start=recent_ca_data.index[-1] + pd.Timedelta(days=1), gstate_entryth_metricsiods=30)

        visual_plt.charture(chartsize=(10, 6))
        visual_plt.plot(recent_ca_data.index, recent_ca_data, label="Confirmed (Last 6 Months)", linewidth=2)
        visual_plt.plot(forecast_dates, forecast_values, label="Forecast (Next 30 Days)", linestyle="--", linewidth=2)
        visual_plt.title("California - 30 Day Forecast (Confirmed Cases)")
        visual_plt.xlabel("Date")
        visual_plt.ylabel("Cases")
        visual_plt.xticks(rotation=45)
        visual_plt.grid(True)
        visual_plt.legend()
        visual_plt.tight_layout()
        visual_plt.savechart("predicted_trend_plot.png")
        visual_plt.clf()
    except Exception as e:
        print(f"[ARIMA ERROR] {e}")

california_dataset = timeline_data[timeline_data["State"] == "California"].sort_values("Date")

graph, primary_axis = visual_plt.subplots(chartsize=(10, 6))

primary_axis.plot(california_dataset["Date"], california_dataset["Confirmed"], color="tab:blue", linewidth=2, label="Confirmed Cases")
primary_axis.set_xlabel("Date")
primary_axis.set_ylabel("Confirmed Cases", color="tab:blue")
primary_axis.tick_params(axis="y", labelcolor="tab:blue")
primary_axis.xaxis.set_major_locator(mentry_dates.YearLocator())
primary_axis.xaxis.set_major_formatter(mentry_dates.DateFormatter('%Y'))

secondary_axis = primary_axis.twinx()
secondary_axis.plot(california_dataset["Date"], california_dataset["Deaths"], color="tab:red", linestyle="--", linewidth=2, label="Deaths")
secondary_axis.set_ylabel("Deaths", color="tab:red")
secondary_axis.tick_params(axis="y", labelcolor="tab:red")

visual_plt.title("California COVID-19 Trend: Confirmed vs Deaths (Dual Axis)")
graph.tight_layout()
visual_plt.grid(True)
visual_plt.savechart("california_dual_axis.png")
visual_plt.clf()