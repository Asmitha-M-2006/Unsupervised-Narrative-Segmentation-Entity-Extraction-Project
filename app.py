import streamlit as st

from src.pipeline import analyze_video


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="News Narrative Segmentation",
    page_icon="📰",
    layout="wide"
)


# --------------------------------------------------
# Application UI
# --------------------------------------------------

st.title("📰 News Narrative Segmentation")

st.write(
    "Automatically segment a long YouTube news video "
    "into narrative stories and extract important entities."
)


video_url = st.text_input(
    "YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)


if st.button("Analyze Video", type="primary"):

    if not video_url:
        st.warning(
            "Please enter a YouTube URL."
        )

    else:

        video_id = (
            video_url
            .split("v=")[-1]
            .split("&")[0]
        )

        with st.spinner(
            "Analyzing video..."
        ):

            result = analyze_video(
                video_id
            )


        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        metadata = result[
            "bulletin_metadata"
        ]

        stories = result[
            "stories"
        ]

        quality = result[
            "segmentation_quality"
        ]


        st.success(
            f"Found {metadata['total_stories']} "
            f"narrative stories."
        )


        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Stories",
                metadata["total_stories"]
            )


        with col2:

            st.metric(
                "Duration",
                f"{metadata['duration_minutes']:.1f} min"
            )


        with col3:

            st.metric(
                "Topics",
                len(
                    quality["topics_identified"]
                )
            )


        st.divider()


        # --------------------------------------------------
        # Story Display
        # --------------------------------------------------

        for story in stories:

            with st.expander(
                f"{story['story_id']} — "
                f"{story['topic']} — "
                f"{story['subtopic']}"
            ):

                # Timestamps are already formatted
                # by pipeline.py as strings such as 1:05.
                st.write(
                    f"⏱️ **{story['timestamp_start']} "
                    f"→ {story['timestamp_end']}**"
                )


                st.write(
                    f"**Topic:** {story['topic']}"
                )


                st.write(
                    f"**Subtopic:** {story['subtopic']}"
                )


                st.write(
                    f"**Confidence:** "
                    f"{story['confidence_score']:.3f}"
                )


                st.write(
                    f"**Words:** "
                    f"{len(story['raw_text'].split())}"
                )


                # --------------------------------------------------
                # Transcript
                # --------------------------------------------------

                st.write("### Transcript")

                st.write(
                    story["raw_text"]
                )


                # --------------------------------------------------
                # Entities
                # --------------------------------------------------

                entities = story["entities"]

                st.write("### Entities")


                for category, entity_list in entities.items():

                    if entity_list:

                        st.write(
                            f"**{category.title()}**"
                        )


                        for entity in entity_list:

                            if "name" in entity:

                                st.write(
                                    f"- {entity['name']}"
                                )

                            elif "value" in entity:

                                st.write(
                                    f"- {entity['value']}"
                                )


                # --------------------------------------------------
                # Keywords
                # --------------------------------------------------

                if story["keywords"]:

                    st.write("### Keywords")

                    st.write(
                        ", ".join(
                            story["keywords"]
                        )
                    )