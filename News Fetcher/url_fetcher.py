import urllib.parse
import datetime
import asyncio
import feedparser
import re
import base64
import aiohttp
import json
import tqdm
import os

RETRY_LIMIT = 3
SIMULTANEOUS_REQUESTS = 50


class URL_FETCHER:
    def __init__(self) -> None:
        self.scrapped_article_details = []
        self.request_success_counter = 0

    def generate_keyword_dicts(self, keywords, language="en", country="US"):
        """_summary_

        Args:
            keywords (_type_): _description_
            language (str, optional): _description_. Defaults to "en".
            country (str, optional): _description_. Defaults to "US".

        Returns:
            _type_: _description_
        """
        keyword_dicts = []
        for keyword in keywords:
            keyword_dicts.append(
                {"language": language, "country": country, "keywords": [keyword]}
            )

        return keyword_dicts

    def base64url_decoder(self, inp, link):
        """
        Args:
            inp: base64 encoded string
            link: article redirecting link; starts with https://news.google.com
        Returns: if the link is decoded successfully then returns the decoded link else returns the input link
        """
        padding_factor = (4 - len(inp) % 4) % 4
        inp += "=" * padding_factor
        decoded_url = base64.urlsafe_b64decode(inp).decode("ISO-8859-1")
        final_link = re.sub(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]",
            "",
            decoded_url[decoded_url.find("http") :],
        )
        tag = final_link.split(":")[0]
        repeaters = [m.start() for m in re.finditer(tag, final_link)]
        if len(repeaters) > 1:
            final_link = final_link[repeaters[1] :]
        if final_link == "":
            return link
        return final_link

    def generate_google_news_url(self, query, when=None, lang="en", country="US"):
        """
        Basic job is to do this:
        https://news.google.com/rss/search?q=%22General+Electric%22+when:24h&ceid=US:en&hl=en-US&gl=US
        Args:
            query: keyword which needs to be scraped
            category_name: category name
            when: duration of data to fetch.. 24h, 1w...
            lang: language; en, de...
            country: US (default)
        Returns:
            final_url: URL to make request
        """
        base_url = "https://news.google.com/rss"
        keyword = query
        if when:
            query += when
        query = urllib.parse.quote_plus(query)
        search_ceid = "&ceid={}:{}&hl={}&gl={}".format(country, lang, lang, country)
        final_url = base_url + "/search?q={}".format(query) + search_ceid
        return final_url, keyword

    def transform_keywords_to_urls(
        self, scraper_keys, start_date=None, end_date=None, timedelta=3
    ):
        final_url_list = []
        query_list = []
        for data in scraper_keys:
            today_date_datetime = start_date
            for _ in range(10000):
                if today_date_datetime < end_date:
                    break

                today_date = "{:%Y-%m-%d}".format(today_date_datetime)
                prior_date_3_datetime = today_date_datetime - datetime.timedelta(
                    days=timedelta
                )
                prior_date_3 = "{:%Y-%m-%d}".format(prior_date_3_datetime)
                query_when = " after:" + prior_date_3 + " before:" + today_date

                # print(query_when)
                result = [
                    self.generate_google_news_url(
                        query=k,
                        lang=data["language"],
                        country=data["country"],
                        when=query_when,
                    )
                    for k in data["keywords"]
                ]
                final_url_list.extend(result)
                query_list.extend([query_when * len(result)])

                today_date_datetime = prior_date_3_datetime

        return final_url_list

    async def make_request_basic(self, url, keyword, retry_counter=RETRY_LIMIT):
        # Hit the URL and get the response without session
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    text = await resp.text()
                    feed = feedparser.parse(text)
                    if resp.status == 200:
                        self.request_success_counter += 1
                    if len(feed["entries"]) > 0:
                        print(
                            "SUCCESS",
                            f"successfully found: {len(feed['entries'])}. ",
                            f"response status code-> {resp.status}",
                            keyword,
                            url,
                        )
                        self.scrapped_article_details.extend(
                            [
                                {
                                    "link": self.base64url_decoder(
                                        data["id"], data["link"]
                                    ),
                                    "keyword": keyword,
                                }
                                for data in feed["entries"]
                            ]
                        )
                await session.close()

        except aiohttp.ClientError as e:
            print(
                "ERROR",
                f"error in fetching data. Error: {e}",
                keyword,
                url,
            )
            if retry_counter > 0:
                await self.make_request_basic(url, keyword, retry_counter - 1)
        except Exception as e:
            print(
                "ERROR",
                f"error in fetching data. Error: {e}",
                keyword,
                url,
            )

    async def create_requests(self, url_list):
        # Create couroutine for each URL
        asyncio.Semaphore(SIMULTANEOUS_REQUESTS)
        tasks = []
        for url, keyword in url_list:
            tasks.append(self.make_request_basic(url, keyword))

        return await asyncio.gather(*tasks)

    def keep_only_unique_links(self):
        # Keep only unique links in self.scrapped_article_details
        unique_links = []
        for data in tqdm.tqdm(self.scrapped_article_details):
            if data["link"] not in unique_links:
                unique_links.append(data["link"])

        self.scrapped_article_details = [
            data
            for data in self.scrapped_article_details
            if data["link"] in unique_links
        ]

    def save_to_json(self, data, filename):
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def main(
        self,
        keywords,
        save_path,
        start_date=None,
        end_date=None,
        timedelta=3,
        langauges=["en"],
        countries=["US"],
        save_json=False,
    ):
        """_summary_

        Args:
            keywords (_type_): _description_
            start_date (_type_, optional): Starting Date. (As recent as possible) Defaults to None.
            end_date (_type_, optional): End date (As late as possible). Defaults to None.
            timedelta (int, optional): _description_. Defaults to 3.
            langauges (list, optional): _description_. Defaults to ['en'].
            countries (list, optional): _description_. Defaults to ['US'].

        Returns:
            _type_: _description_
        """

        if not start_date:
            start_date = datetime.datetime.now()
            end_date = datetime.datetime.now() - datetime.timedelta(days=timedelta)

        keyword_dicts = []

        for langauge, country in zip(langauges, countries):
            keyword_dicts.extend(
                self.generate_keyword_dicts(
                    keywords, language=langauge, country=country
                )
            )
        final_url_list = self.transform_keywords_to_urls(
            keyword_dicts, start_date, end_date, timedelta
        )

        asyncio.run(self.create_requests(final_url_list))
        print(f"Total requests made: {len(self.scrapped_article_details)}")
        self.keep_only_unique_links()
        print(f"Total unique links found: {len(self.scrapped_article_details)}")

        if save_json:
            save_filename = f"{save_path}/scrapped_raw_url.json"
            self.save_to_json(self.scrapped_article_details, save_filename)
            print(f"Data saved to {save_filename}")

        return self.scrapped_article_details


if __name__ == "__main__":
    url_fetcher = URL_FETCHER()
    now_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    save_path = "./data/run_" + now_datetime
    os.makedirs(save_path, exist_ok=True)
    url_fetcher.main(
        keywords=["General Electric", "Apple"],
        start_date=datetime.datetime(2024, 3, 1),
        end_date=datetime.datetime(2024, 2, 27),
        timedelta=3,
        langauges=["en"],
        countries=["US"],
        save_json=True,
        save_path=save_path,
    )
