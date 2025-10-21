run:
	python run.py

report:
	python run.py

dashboard:
	streamlit run app/streamlit_app.py

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache
