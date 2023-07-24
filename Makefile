CODE=guidedsum
TESTS=tests

format:
	black ${CODE} ${TESTS} scripts
	isort ${CODE} ${TESTS} scripts

test:
	pytest --cov-report html --cov=${CODE} ${CODE} ${TESTS}

lint:
	pylint --disable=R,C ${CODE} ${TESTS}
	black --check ${CODE} ${TESTS}
	isort --check-only ${CODE} ${TESTS}

lintci:
	pylint --disable=W,R,C ${CODE} ${TESTS}
	black --check ${CODE} ${TESTS}
	isort --check-only ${CODE} ${TESTS}
