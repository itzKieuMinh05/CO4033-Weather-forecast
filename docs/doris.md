-- tránh lỗi session
SWITCH internal;

-- xóa catalog cũ nếu có
DROP CATALOG IF EXISTS iceberg;

-- tạo lại Iceberg catalog
CREATE CATALOG iceberg PROPERTIES (
    "type" = "iceberg",
    "iceberg.catalog.type" = "rest",
    "uri" = "http://iceberg-rest:8181",

    "s3.endpoint" = "http://minio:9000",
    "s3.access_key" = "minioadmin",
    "s3.secret_key" = "minioadmin123",
    "s3.region" = "us-east-1",

    "use_path_style" = "true",
    "s3.connection.ssl.enabled" = "false"
);