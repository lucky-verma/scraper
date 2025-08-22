# 🚀 Web Scraper

A FastAPI-based intelligent web scraper powered by **crawl4ai** with advanced business content processing and caching.

## ✨ Features

- **Smart Web Scraping** - Powered by crawl4ai with Playwright
- **Business Intelligence** - Extracts emails, phone numbers, executives, financials
- **File Processing** - Converts PDFs, Word docs, Excel files to markdown
- **Content Optimization** - Dynamic token limiting and content prioritization
- **PostgreSQL Caching** - 14-day TTL with intelligent URL canonicalization
- **Content Classification** - Auto-detects business websites, news, technical docs

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/lucky-verma/scraper
cd scraper

# Docker Setup
docker build -t scraper .
docker run -p 8081:8081 scraper
```

## 📡 API Endpoints

### Scrape Web Page

```bash
POST /crawl
{
  "url": "https://example.com"
}
```

### Health Check

```bash
GET /health
```

### Cache Statistics

```bash
GET /cache/stats
```

## 🎯 Use Cases

- **Company Profiling** - Extract business information from corporate websites
- **Lead Generation** - Gather contact details and executive information  
- **Market Research** - Process competitor websites and industry reports
- **Document Processing** - Convert and analyze PDF reports and presentations

## 🏗️ Architecture

- **FastAPI** - High-performance async web framework
- **crawl4ai** - Advanced web crawling with Playwright
- **PostgreSQL** - Persistent caching and metadata storage
- **TikToken** - Intelligent content tokenization and optimization

***

**Status**: Production Ready | **Port**: 8081 | **Cache TTL**: 14 days
