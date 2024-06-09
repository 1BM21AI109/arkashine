mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"sahana.ai21@bmsce.ac.in\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
