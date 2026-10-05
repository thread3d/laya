// `laya-java` lives in the laya monorepo so `scripts/gen_fixtures.py` can run the real Python in
// the same checkout. That is what makes the parity gate possible without a submodule or a
// published artifact to measure against.
rootProject.name = "laya-java-root"

include("laya-java", "laya-java-client")
