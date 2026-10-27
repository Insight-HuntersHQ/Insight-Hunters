from helpers import *
import streamlit as st


st.set_page_config(
    page_title="Insight Hunters - TB Dashboard",  
    page_icon="page_icon.png",                                
    layout="wide"                                  
)
pages = [
        st.Page(main_page, title =  "Home", icon=":material/home:"),
        st.Page(tb_cat_page, title = "Model", icon= ":material/conditions:"),
        st.Page(info_page, title = "Info", icon = ":material/info:")
    ]

current_page = st.navigation(pages, position= "sidebar")

current_page.run()