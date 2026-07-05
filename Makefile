.PHONY: help check-nvidia up down restart logs status pull clean

help:
	@echo "make up      — start all GPU services"
	@echo "make down    — stop all services"
	@echo "make restart — down + up"
	@echo "make logs    — tail logs"
	@echo "make status  — container status"
	@echo "make pull    — git pull + rebuild + restart"
	@echo "make clean   — remove containers, images, volumes"

check-nvidia:
	@which nvidia-container-toolkit > /dev/null || \
		(echo "nvidia-container-toolkit not found — required for GPU passthrough" && exit 1)

up: check-nvidia
	docker compose up -d

down:
	docker compose down

restart: down up

logs:
	docker compose logs -f

status:
	docker compose ps

pull:
	git pull
	docker compose up -d --build

clean:
	docker compose down --rmi local -v
