"""Scrap article texts from a website."""

import argparse
import csv
import re
from os import path
from time import sleep
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

FIELDNAMES = ["id", "title", "body"]
SEPARATOR = " "
VERBOSE = False


class Parser:
    """Parse web pages.

    Attributes:
        article_regex (str): Regex to get article links.
        page_regex (str): Regex to get links to other pages with article links.
        id_regex (str): Regex to get article id.
        title_tag (str): HTML tag used for article's title.
        body_class (str): CSS class that marks articles text.
    """

    def __init__(
        self,
        article_regex: str,
        page_regex: str,
        id_regex: str = r"\d+",
        title_tag: str = "h1",
        body_class: str = "article-formatted-body",
    ):
        self.article_regex = article_regex
        self.page_regex = page_regex
        self.id_regex = id_regex
        self.title_tag = title_tag
        self.body_class = body_class

    def parse_article(self, article: str) -> (str, str):
        """Get article title and text.

        Args:
            article: Article url.
        """

        response = requests.get(article)
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.find(self.title_tag).text
        title = title.replace("\xa0", " ")
        body = soup.find(class_=self.body_class).get_text(SEPARATOR)
        body = body.replace("\xa0", " ")

        return title, body

    def parse_page(self, url: str) -> (list[str], list[str]):
        """Get links to articles and other pages with articles from a page."""

        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a")

        article_pattern = re.compile(self.article_regex)
        page_pattern = re.compile(self.page_regex)

        articles = []
        pages = []
        for link in links:
            href = link.get("href")
            if article_pattern.match(str(href)):
                articles.append(href)
            elif page_pattern.match(str(href)):
                pages.append(href)

        return articles, pages

    def article_id(self, article: str):
        """Get article id."""
        return re.findall(self.id_regex, article)[-1]


class Crawler:
    """Retrieve and save articles from a web-site.

    Attributes:
        parser (Parser): Parser used to retrieve data.
        data_csv (str): CSV to save article data to.
        crawl_delay (str): Minimal time to wait between HTTP requests, see robots.txt.
    """

    def __init__(
        self,
        parser: Parser,
        data_csv: str,
        crawl_delay: int = 10,
    ):
        self.visited_articles = []
        self.visited_pages = []
        self.data = data_csv

        if path.exists(self.data):
            with open(self.data, "r") as f:
                assert f.readlines()[0].startswith("id,title,body")
            with open(self.data, "r") as f:
                reader = csv.DictReader(f)
                self.visited_articles = [row["id"] for row in reader]
        else:
            with open(self.data, "w") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()

        self.parser = parser
        self.crawl_delay = crawl_delay

    def crawl(self, url: str) -> None:
        """Traverse webpage, looking for article links."""

        if VERBOSE:
            print(f"Crawling {url}")

        articles, pages = self.parser.parse_page(url)

        for article in articles:
            id = self.parser.article_id(article)
            if id in self.visited_articles:
                continue

            sleep(self.crawl_delay)
            article_url = absolute_url(article, url)
            if VERBOSE:
                print(f"Parsing {article_url}")
            title, body = self.parser.parse_article(article_url)
            self.save_article(id, title, body)

        for page in pages:
            page_no = int(re.findall(r"\d+", page)[-1])
            if page_no in self.visited_pages:
                continue

            sleep(self.crawl_delay)
            self.visited_pages.append(page_no)
            page_url = absolute_url(page, url)
            self.crawl(page_url)

    def save_article(self, id, title, body) -> None:
        """Save retrieved article."""
        with open(self.data, "a") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writerow({"id": id, "title": title, "body": body})
        self.visited_articles.append(id)


def absolute_url(rel_url: str, ref_url: str) -> str:
    """Build an absolute url from a relative url and a reference absolute url.

    Args:
        rel_url (str): relative url to convert to absolute
        ref_url (str): absolute url to get base url from
    """
    split_url = urlsplit(ref_url)
    base_url = "".join([split_url.scheme, "://", split_url.netloc])
    return base_url + rel_url


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog="scrap",
        description=__doc__,
    )

    argparser.add_argument("data_csv", type=str)
    argparser.add_argument("-u", "--url", type=str, required=True)
    argparser.add_argument("-a", "--article-regex", type=str, required=True)
    argparser.add_argument("-p", "--page-regex", type=str, required=True)
    argparser.add_argument("-i", "--id", "--id-regex", type=str, default=r"\d+")
    argparser.add_argument("-t", "--title-tag", type=str, default="h1")
    argparser.add_argument(
        "-b", "--body-class", type=str, default="article-formatted-body"
    )
    argparser.add_argument("-d", "--delay", "--crawl-delay", type=int, default=10)
    argparser.add_argument("-v", "--verbose", action="store_true")

    args = argparser.parse_args()

    VERBOSE = args.verbose

    parser = Parser(
        args.article_regex,
        args.page_regex,
        args.id,
        args.title_tag,
        args.body_class,
    )

    crawler = Crawler(
        parser,
        args.data_csv,
        args.delay,
    )

    crawler.crawl(args.url)
