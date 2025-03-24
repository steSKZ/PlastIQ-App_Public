import streamlit as st

st.write("<h2 style='text-align: center;'>System zur Prognose und Verwertungssteuerung von Kunststoffabfällen</h2>", unsafe_allow_html=True)

c1, c2 = st.columns([2, 3])
with c1:
    st.write("### Hintergrund")
    st.text("""Bei vielen Unternehmen fallen Kunststoffabfälle unterschiedlichster Art an. Vielfach werden diese minderwertig verwertet, da Informationsdefizite hinsichtlich deren Qualität, Zusammensetzung und Verfügbarkeit sowie potentieller Abnehmer und möglicher Verwertungsoptionen bestehen. """)
with c2:
    st.write("### Über")
    st.text("""
    PlastIQ zielt daher als ein System zur Bewertung anfallender Kunststoffabfälle darauf ab diese Lücke zu schließen. Die Prognose von Menge, Qualität und Verfügbarkeit der Abfälle sowie ein Matching mit potentiellen Abnehmern und Verwertungsoptionen ermöglicht es den Unternehmen, eine optimale Verwertung ihrer Abfälle zu erreichen und dadurch die Kreislaufführung von Kunststoffmaterialen zu befördern.
     
    Die Plattform wurde von dem SKZ und WeSort.AI mit der Unterstützung des Bayrischen Ministerium entwickelt.
    """)


with st.columns(3)[1]:
    if st.button("**🚀 Los geht's!**", use_container_width=True):
        st.switch_page("subpages/input/contact.py")
