import nox


@nox.session
def tests(session):
    session.install("-e", ".[test]")
    session.run("pytest", "-v", "tests")


@nox.session
def lint(session):
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")
