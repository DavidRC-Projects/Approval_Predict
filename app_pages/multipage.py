import streamlit as st


class MultiPage:

    def __init__(self, app_name) -> None:
        """Class for managing a multi-page Streamlit application."""
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🏦")

    def add_page(self, title, func) -> None:
        """
        Initialize the MultiPage manager.
        """
        self.pages.append({"title": title, "function": func})

    def run(self):
        """
        Add a page to the app.
        """
        st.title(self.app_name)
        page = st.sidebar.radio(
            'Menu',
            self.pages,
            format_func=lambda page: page['title']),
        page['function']()
