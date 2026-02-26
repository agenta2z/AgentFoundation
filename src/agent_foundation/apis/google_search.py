import json
from pprint import pprint
from typing import Union, Iterable, Dict, List, Tuple

import requests

from rich_python_utils.common_utils import iter_, resolve_environ
from rich_python_utils.console_utils import hprint_message
from rich_python_utils.string_utils import join_

ENV_NAME_GOOGLE_SEARCH_APIKEY = 'GOOGLE_SEARCH_APIKEY'
ENV_NAME_GOOGLE_CSE_ID = 'GOOGLE_CSE_ID'
API_URL_GOOGLE_SEARCH = 'https://www.googleapis.com/customsearch/v1'


def google_search(
        search_term: str,
        start_date: str = None,
        end_date: str = None,
        sites: Union[str, Iterable[str]] = None,
        api_key: str = None,
        cse_id: str = None,
        verbose: bool = True,
        return_raw_results: bool = False,
        **extra_constraints
) -> Union[Dict, List[Tuple[str, str, str]]]:
    """
    Performs a Google search using the Custom Search JSON API and retrieves search results
    based on specified constraints.

    Args:
        search_term (str): The primary term to search for.
        start_date (str, optional): Date in the format 'YYYY-MM-DD' to start the search from.
            Defaults to None.
        end_date (str, optional): Date in the format 'YYYY-MM-DD' to end the search. Defaults to None.
        sites (Union[str, Iterable[str]], optional): Site(s) to restrict the search to. Accepts a single
            site or an iterable of site strings. Defaults to None.
        api_key (str, optional): Google API key. If None, uses the environment variable
            `GOOGLE_SEARCH_APIKEY`. Defaults to None.
        cse_id (str, optional): Custom Search Engine ID. If None, uses the environment variable
            `GOOGLE_CSE_ID`. Defaults to None.
        verbose (bool, optional): If True, prints the constructed search term. Defaults to True.
        return_raw_results (bool, optional): If True, returns the raw API JSON response. If False,
            returns a list of tuples containing URL, title, and snippet of each search result. Defaults to False.
        **extra_constraints: Additional keyword arguments for specific search constraints (e.g., "filetype:pdf").

    Returns:
        Union[Dict, List[Tuple[str, str, str]]]: Either the raw JSON response (if `return_raw_results`
        is True) or a list of tuples with each tuple containing the URL, title, and snippet of each
        result.

    Raises:
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    api_key = resolve_environ(api_key or ENV_NAME_GOOGLE_SEARCH_APIKEY)
    cse_id = resolve_environ(cse_id or ENV_NAME_GOOGLE_CSE_ID)

    # add time constraints to search term
    search_term = join_(
        (
            search_term,
            ("after:" + start_date) if start_date else None,
            ("before:" + end_date) if end_date else None,
            *((("site:" + site) for site in iter_(sites)) if sites else ()),
            *((f"{k}:{v}" for k, v in extra_constraints.items()))
        ), sep=' ')

    if verbose:
        hprint_message(
            'search_term', search_term
        )

    # The query parameters
    params = {
        'q': search_term,
        'key': api_key,
        'cx': cse_id
    }

    # Make a GET request to the API
    response = requests.get(API_URL_GOOGLE_SEARCH, params=params)

    # Parse the JSON response
    search_results = json.loads(response.text)

    if return_raw_results:
        return search_results
    else:
        if 'items' in search_results:
            return [
                (item['link'], item['title'], item['snippet'])
                for item in search_results['items']
            ]
        else:
            return []


if __name__ == '__main__':
    from rich_python_utils.common_utils.arg_utils.arg_parse import get_parsed_args
    args = get_parsed_args(
        default_search_term='AAPL stock top stories',
        default_start_date='',
        default_end_date='',
        default_sites='[]',
    )

    _search_term = args.search_term
    _start_date = args.start_date
    _end_date = args.end_date
    _sites = args.sites

    pprint(
        google_search(
            search_term=_search_term,
            start_date=_start_date,
            end_date=_end_date,
            sites=_sites
        )
    )
