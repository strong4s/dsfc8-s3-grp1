import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import math
from streamlit_folium import folium_static
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import Normalizer, MinMaxScaler
from math import pi
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image

pd.set_option('display.float_format', '{:,.4f}'.format)
pd.set_option('max_columns', 100)

# * Importing Data, Data Cleaning | Can upload multiple DFs
# df_BB_TracksData=pd.read_csv("/home/fqp/Documents/0 Study/3 Data Science Cohort 8_Eskwelabs/Sprint 3 Folder/BenBen_playlist_tracks_data.csv")
# df_spot2 = pd.read_csv('/home/fqp/Documents/0 Study/3 Data Science Cohort 8_Eskwelabs/Sprint 3 Folder/spotify_tracks_withgenre.csv')
# df_spot = pd.read_csv('/home/fqp/Documents/0 Study/3 Data Science Cohort 8_Eskwelabs/Sprint 3 Folder/dsf-c8-sprint3/data/spotify_daily_charts_tracks_rec_pool.csv')
df_hope_bb = pd.read_csv('data/hopeful_df.csv')
df_sawi_bb = pd.read_csv('data/sawi_df.csv')

df_hope_EUC = pd.read_csv('data/hope_EUC.csv')
df_hope_COS = pd.read_csv('data/hope_COS.csv')
df_sawi_EUC = pd.read_csv('data/sawi_EUC.csv')
df_sawi_COS = pd.read_csv('data/sawi_COS.csv')

# Radio buttons = Table of Contents
# ? Workaround for no default selected radio button.
st.markdown(""" <style>
            div[role="radiogroup"] >  :first-child{
                display: none !important;
            }
        </style>
                """, unsafe_allow_html=True)
st.title("A Ben&Ben Collaboration Playlist Recommender")
st.write("Create a playlist featuring Ben&Ben and other OPM artists by defining the mood, and how similar are their features.")
# st.subheader("Sprint 3 | Group 1")

page_selection = st.sidebar.radio(label="How are you feeling today?",options=["","Hopeful","Sawi"])

if page_selection == 'Hopeful':
    st.sidebar.subheader("Further define your playlist through the sidebar and the main page prompts.")
    
    #? MOOD
    sim_mes = st.sidebar.selectbox(label="Similarity Measure",options=["","Euclidean","Cosine"])
    if sim_mes == "Euclidean":
        # * Sliders
        df_hope_EUC["euclidean_dist"] = -df_hope_EUC["euclidean_dist"]
        h_e_min = float(df_hope_EUC["euclidean_dist"].min())
        h_e_max = float(df_hope_EUC["euclidean_dist"].max())
        dist_min = st.sidebar.slider("Similarity to Ben&Ben", min_value=h_e_min, max_value=h_e_max,step=0.005)
            
        h_v_min = float(df_hope_EUC["valence"].min())
        h_v_max = float(df_hope_EUC["valence"].max())
        valence_slider = st.sidebar.slider("Emotional Positivity",min_value=h_v_min,max_value=h_v_max)#,step=0.005)
            
        h_a_min = float(df_hope_EUC["acousticness"].min())
        h_a_max = float(df_hope_EUC["acousticness"].max())
        acc_slider = st.sidebar.slider("Acousticness",min_value=h_a_min,max_value=h_a_max)#,step=0.005)
        ###### *  
        
        st.subheader("These are Ben&Ben's hopeful songs")
        df_hope_bb[["track_name","artist_name"]]
        
        # ? Artists Filter
        filter = st.radio(label="Do you want to manually filter the collaboration artists?",options=["","Yes","No"])
        if filter == "Yes":
            st.subheader('These are other hopeful songs from prospect collaboration artists')
            df_hope_EUC[["track_name","artist_name"]]
            # 'danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo'
            artists_onpage = df_hope_EUC["artist_name"].unique()
            artists_mask = st.multiselect("Filter with the artists you want",options=artists_onpage)
            st.header("Here is the recommended tracks pool")
            df_out = df_hope_EUC.loc[(df_hope_EUC["artist_name"].isin(artists_mask)) & (df_hope_EUC["euclidean_dist"] >= dist_min) & (df_hope_EUC["valence"] >= valence_slider) & (df_hope_EUC["acousticness"] >= acc_slider)][["track_name","artist_name",'danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']]
            df_out
            
        elif filter == "No":
            st.header("Here is the recommended tracks pool")        
            df_out = df_hope_EUC.loc[(df_hope_EUC["euclidean_dist"] >= dist_min) & (df_hope_EUC["valence"] >= valence_slider) & (df_hope_EUC["acousticness"] >= acc_slider)][["track_name","artist_name",'danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']]      
            df_out
        
        
        # ? Button for GENERATING the Playlist
        if st.button('GENERATE PLAYLIST'):
            st.header("Here is your randomized playlist with Ben & Ben tracks!")
            # df_out [index, track_name, artist_name]
            rand = np.random.choice(df_hope_bb.index, math.ceil(len(df_out.index)/2)+3)
            df_bb_out = df_hope_bb.loc[rand,["track_name","artist_name",'danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']]
            df_bb_out.drop_duplicates(inplace=True)
            df_recomm = pd.concat([df_bb_out,df_out])
            df_recomm = df_recomm.sample(frac=1).reset_index(drop=True)
            df_recomm
            st.write("You may press the GENERATE PLAYLIST button again, or re-define your preferences.")
        else:
            ""
        
         # *

    
    # ? Hopeful - Cosine                   
    elif sim_mes == "Cosine":
        # * Sliders
        df_hope_COS["cosine_dist"] = -df_hope_COS["cosine_dist"]
        h_c_min = float(df_hope_COS["cosine_dist"].min())
        h_c_max = float(df_hope_COS["cosine_dist"].max())
        dist_min = st.sidebar.slider("Similarity to Ben&Ben",min_value=h_c_min,max_value=h_c_max,step=0.0025)
            
        h_v_min = float(df_hope_COS["valence"].min())
        h_v_max = float(df_hope_COS["valence"].max())
        valence_slider = st.sidebar.slider("Emotional Positivity",min_value=h_v_min,max_value=h_v_max)#,step=0.005)
            
        h_a_min = float(df_hope_COS["acousticness"].min())
        h_a_max = float(df_hope_COS["acousticness"].max())
        acc_slider = st.sidebar.slider("Acousticness",min_value=h_a_min,max_value=h_a_max)#,step=0.005)
        
        st.subheader("These are Ben&Ben's hopeful songs")
        df_hope_bb[["track_name","artist_name"]]
        
         
        filter = st.radio(label="Do you want to filter artists?",options=["","Yes","No"])
        if filter == "Yes":                    
            st.subheader('These are other hopeful songs from prospect collaboration artists')
            df_hope_COS[["track_name","artist_name"]]
            
            artists_onpage = df_hope_COS["artist_name"].unique()
            artists_mask = st.multiselect("Filter with the artists you want",options=artists_onpage)
            st.header("Here is the tracks recommendation pool")
            df_out = df_hope_COS.loc[(df_hope_COS["artist_name"].isin(artists_mask)) & (df_hope_COS["cosine_dist"] >= dist_min) & (df_hope_COS["valence"] >= valence_slider) & (df_hope_COS["acousticness"] >= acc_slider)][["track_name","artist_name",'danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']]
            df_out
             # *
            
        elif filter == "No":
            st.header("Preview your playlist here")
            df_out = df_hope_COS.loc[ (df_hope_COS["cosine_dist"] >= dist_min) & (df_hope_COS["valence"] >= valence_slider) & (df_hope_COS["acousticness"] >= acc_slider)][["track_name","artist_name",'danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']]
            df_out
            
        
        if st.button('GENERATE PLAYLIST'):
            st.header("Here is your randomized playlist with Ben & Ben tracks!")
            rand = np.random.choice(df_hope_bb.index, math.ceil(len(df_out.index)/2)+3, replace=False)
            df_bb_out = df_hope_bb.loc[rand,["track_name","artist_name",'danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']]
            df_recomm = pd.concat([df_bb_out,df_out])
            df_recomm = df_recomm.sample(frac=1).reset_index(drop=True)
            df_recomm
            st.write("You may press the GENERATE PLAYLIST button again, or re-define your preferences.")
        
        else:
            ""
            # st.sidebar.
            # st.dataframe(my_dataframe)
        # #  ? FILTER button
        # if st.button("Reset") == True:
        #     df_recom = df_hope_EUC.loc[(df_hope_EUC["artist_name"].isin(artists_mask)) & (df_hope_EUC["euclidean_dist"] >= dist_min) & (df_hope_EUC["valence"] >= valence_slider) & (df_hope_EUC["acousticness"] >= acc_slider)][["track_name","artist_name"]]

elif page_selection == "Sawi":
    st.sidebar.subheader("Further define your playlist by through the sidebar, and with the main page prompts.")
    sim_mes = st.sidebar.selectbox(label="Similarity Measure",options=["","Euclidean","Cosine"])
    
    
    if sim_mes == "Euclidean":
        df_sawi_EUC["euclidean_dist"] = -df_sawi_EUC["euclidean_dist"]
        h_e_min = float(df_sawi_EUC["euclidean_dist"].min())
        h_e_max = float(df_sawi_EUC["euclidean_dist"].max())
        dist_min = st.sidebar.slider("Similarity to Ben&Ben",min_value=h_e_min,max_value=h_e_max,step=0.005)
             
        h_v_min = float(df_sawi_EUC["valence"].min())
        h_v_max = float(df_sawi_EUC["valence"].max())
        valence_slider = st.sidebar.slider("Emotional Positivity",min_value=h_v_min,max_value=h_v_max)#,step=0.005)
            
        h_a_min = float(df_sawi_EUC["acousticness"].min())
        h_a_max = float(df_sawi_EUC["acousticness"].max())
        acc_slider = st.sidebar.slider("Acousticness",min_value=h_a_min,max_value=h_a_max)#,step=0.005)
        ###### *  
        
        st.subheader("These are Ben&Ben's sawi songs")
        df_sawi_bb[["track_name","artist_name"]]
        
        # ? Artists Filter
        filter = st.radio(label="Do you want to manually filter the other artists?",options=["","Yes","No"])
        if filter == "Yes":
            st.subheader('These are other hopeful songs from prospect collaboration artists')
            df_sawi_EUC[["track_name","artist_name"]]
            # 'danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo'
            artists_onpage = df_sawi_EUC["artist_name"].unique()
            artists_mask = st.multiselect("Filter with the artists you want",options=artists_onpage)
            st.header("Here is the tracks recommendation pool")
            df_out = df_sawi_EUC.loc[(df_sawi_EUC["artist_name"].isin(artists_mask)) & (df_sawi_EUC["euclidean_dist"] >= dist_min) & (df_sawi_EUC["valence"] >= valence_slider) & (df_sawi_EUC["acousticness"] >= acc_slider)][["track_name","artist_name",'danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']]
            df_out
            
        elif filter == "No":
            st.header("Here is the tracks recommendation pool")        
            df_out = df_sawi_EUC.loc[(df_sawi_EUC["euclidean_dist"] >= dist_min) & (df_sawi_EUC["valence"] >= valence_slider) & (df_sawi_EUC["acousticness"] >= acc_slider)][["track_name","artist_name",'danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']]      
            df_out
        
          
        # ? Button for GENERATING the Playlist
        if st.button('GENERATE PLAYLIST'):
            st.header("Here is your randomized playlist with Ben & Ben tracks!")
            # df_out [index, track_name, artist_name]
            rand = np.random.choice(df_sawi_bb.index, math.ceil(len(df_out.index)/2)+3, replace=False)
            df_bb_out = df_sawi_bb.loc[rand,["track_name","artist_name",'danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']]
            # df_bb_out
            df_recomm = pd.concat([df_bb_out,df_out])
            df_recomm = df_recomm.sample(frac=1).reset_index(drop=True)
            df_recomm
            st.write("You may press the GENERATE PLAYLIST button again, or re-define your preferences.")
        else:
            ""
    
    elif sim_mes == "Cosine":
        # * Sliders
        df_sawi_COS["cosine_dist"] = -df_sawi_COS["cosine_dist"]
        h_c_min = float(df_sawi_COS["cosine_dist"].min())
        h_c_max = float(df_sawi_COS["cosine_dist"].max())
        dist_min = st.sidebar.slider("Similarity to Ben&Ben",min_value=h_c_min,max_value=h_c_max,step=0.0025)
            
        h_v_min = float(df_sawi_COS["valence"].min())
        h_v_max = float(df_sawi_COS["valence"].max())
        valence_slider = st.sidebar.slider("Emotional Positivity",min_value=h_v_min,max_value=h_v_max)#,step=0.005)
            
        h_a_min = float(df_sawi_COS["acousticness"].min())
        h_a_max = float(df_sawi_COS["acousticness"].max())
        acc_slider = st.sidebar.slider("Acousticness",min_value=h_a_min,max_value=h_a_max)#,step=0.005)
        
        st.subheader("These are Ben&Ben's sawi songs")
        df_sawi_bb[["track_name","artist_name"]]
        
         
        filter = st.radio(label="Do you want to filter artists?",options=["","Yes","No"])
        if filter == "Yes":                    
            st.subheader('These are other hopeful songs from prospect collaboration artists')
            df_sawi_COS[["track_name","artist_name"]]
            
            artists_onpage = df_sawi_COS["artist_name"].unique()
            artists_mask = st.multiselect("Filter with the artists you want",options=artists_onpage)
            st.header("Here is the tracks recommendation pool")
            df_out = df_sawi_COS.loc[(df_sawi_COS["artist_name"].isin(artists_mask)) & (df_sawi_COS["cosine_dist"] >= dist_min) & (df_sawi_COS["valence"] >= valence_slider) & (df_sawi_COS["acousticness"] >= acc_slider)][["track_name","artist_name",'danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']]
            df_out

        # ! ONGOING   
        elif filter == "No":
            st.header("Preview your playlist here")
            df_out = df_sawi_COS.loc[ (df_sawi_COS["cosine_dist"] >= dist_min) & (df_sawi_COS["valence"] >= valence_slider) & (df_sawi_COS["acousticness"] >= acc_slider)][["track_name","artist_name",'danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']]
            df_out
            
        
        if st.button('GENERATE PLAYLIST'):
            st.header("Here is your randomized playlist with Ben & Ben tracks!")
            rand = np.random.choice(df_sawi_bb.index, math.ceil(len(df_out.index)/2)+3, replace=False)
            df_bb_out = df_hope_bb.loc[rand,["track_name","artist_name",'danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo']]
            df_recomm = pd.concat([df_bb_out,df_out])
            df_recomm = df_recomm.sample(frac=1).reset_index(drop=True)
            df_recomm
            st.write("You may press the GENERATE PLAYLIST button again, or re-define your preferences.")
        else:
            ""
