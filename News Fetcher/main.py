from url_fetcher import URL_FETCHER
from data_fetcher import DataFetcher
from utils import TimeElapsed
import datetime
import os
import json


def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def main(
    keywords,
    start_date=None,
    end_date=None,
    timedelta=3,
    langauges=["en"],
    countries=["US"],
):

    if not start_date:
        start_date = datetime.datetime.now()
        end_date = datetime.datetime.now() - datetime.timedelta(days=timedelta)

    now_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    save_path = "./data/run_" + now_datetime
    os.makedirs(save_path, exist_ok=True)

    metadata = {
        "keywords": keywords,
        "start_date": start_date.strftime("%Y-%m-%d") if start_date else None,
        "end_date": end_date.strftime("%Y-%m-%d") if end_date else None,
        "timedelta": timedelta,
        "langauges": langauges,
        "countries": countries,
        "save_path": save_path,
    }

    save_json(metadata, f"{save_path}/metadata.json")

    time_elapsed = TimeElapsed()
    url_fetcher = URL_FETCHER()
    article_urls = url_fetcher.main(
        keywords=keywords,
        start_date=start_date,
        end_date=end_date,
        timedelta=timedelta,
        langauges=langauges,
        countries=countries,
        save_json=True,
        save_path=save_path,
    )
    url_time = time_elapsed.get_time_elapsed()
    print("Time taken to fetch URLs: ", url_time)

    data_fetcher = DataFetcher()
    data_fetcher.main(
        article_urls=article_urls, save_json=True, save_csv=True, save_path=save_path
    )
    total_time = time_elapsed.get_time_elapsed()
    print("Time taken to fetch data: ", total_time - url_time)
    print("Total time taken: ", total_time)


if __name__ == "__main__":
    main(
        keywords=["Apple"],
        start_date=datetime.datetime(2024, 3, 1),
        end_date=datetime.datetime(2024, 3, 1),
        timedelta=3,
        langauges=["en"],
        countries=["US"],
    )
