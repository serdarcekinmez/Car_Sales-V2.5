#!/usr/bin/env bash
# Garaj demo — Ubuntu / WSL başlatıcı.
# .env dosyasını yükler, sunucuyu yerel modda başlatır ve tarayıcıyı açar.
# Sayfadaki "Oturumu kapat" düğmesi sunucuyu durdurur; bu betik de biter ve
# terminal penceresi (betik çift tıklamayla / "terminalde çalıştır" ile açıldıysa) kapanır.
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
if [ ! -f .env ]; then
  echo "baslat.sh: .env bulunamadı. Önce: cp .env.example .env (ve kullanıcı adı / parola özetini doldurun)" >&2
  exit 2
fi
set -a
# shellcheck disable=SC1091
source .env
set +a
exec python pipeline/run_demo.py serve --insecure-local-http --allow-shutdown --open-browser "$@"
