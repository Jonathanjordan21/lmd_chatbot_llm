#!/bin/bash
set -e

# This script will be executed during the initialization of the PostgreSQL container

echo "Creating lmd_db..."
psql -U postgres -a -f /docker-entrypoint-initdb.d/init1.sql

echo "Restoring tables to lmd_db from SQL dumps"
psql -U postgres -d lmd_db -a -f /docker-entrypoint-initdb.d/init.sql
