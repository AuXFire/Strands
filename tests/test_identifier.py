from strands.identifier import split_identifier


def test_camel_case():
    assert split_identifier("fetchUserData") == ["fetch", "user", "data"]


def test_pascal_case():
    assert split_identifier("HttpResponseCode") == ["http", "response", "code"]


def test_snake_case():
    assert split_identifier("get_active_users") == ["get", "active", "users"]


def test_screaming_snake():
    assert split_identifier("MAX_RETRY_COUNT") == ["maximum", "retry", "count"]


def test_kebab_case():
    assert split_identifier("user-profile-api") == ["user", "profile", "interface"]


def test_abbreviation_expansion():
    assert split_identifier("getDBConn") == ["get", "database", "connection"]
    assert split_identifier("reqMsg") == ["request", "message"]


def test_acronym_inside_pascal():
    # PascalCase with acronym block: HTTPResponseCode → http response code
    assert split_identifier("HTTPResponseCode") == ["http", "response", "code"]


def test_with_digits():
    assert split_identifier("user42Name") == ["user", "42", "name"]


def test_empty():
    assert split_identifier("") == []


def test_single_lowercase_word():
    assert split_identifier("response") == ["response"]
