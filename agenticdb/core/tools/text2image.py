import uuid
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.utils.logger import ModernLogger

# 配置图片存储目录
STATIC_DIR = Path(__file__).parent.parent.parent / "static"
IMAGES_DIR = STATIC_DIR / "images"

# 确保目录存在
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


class Text2ImageAPI(ModernLogger):
    def __init__(self, base_url: str = "https://api.aimlapi.com/v2"):
        super().__init__(name="Text2ImageAPI")
        self.api_key = "49cbbc30ad784ca9a62d416db2a50bff"
        self.base_url = base_url

    def generate_text2image(
        self,
        prompt: str,
        size: str = "1024x1024",
        save_locally: bool = True
    ) -> dict:
        """
        生成图片并可选择保存到本地

        Args:
            prompt: 图片描述
            size: 图片尺寸
            save_locally: 是否保存到本地（默认 True）

        Returns:
            dict: {
                "success": bool,
                "url": str,  # 外部可访问的 URL
                "original_url": str,  # API 返回的原始 URL
                "filename": str,  # 文件名（如果保存到本地）
                "local_path": str  # 本地路径（如果保存到本地）
            }
        """
        try:
            response = requests.post(
                "https://api.aimlapi.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "prompt": prompt,
                    "model": "bytedance/seedream-v4-text-to-image",
                    "size": size,
                    "watermark": False
                }
            )

            response.raise_for_status()
            data = response.json()
            self.info(f"Generation: {data}")

            original_url = data["data"][0]["url"]

            if save_locally:
                # 下载并保存图片到本地
                result = self._download_and_save(original_url)
                result["original_url"] = original_url
                return result
            else:
                return {
                    "success": True,
                    "url": original_url,
                    "original_url": original_url,
                    "filename": None,
                    "local_path": None
                }

        except requests.RequestException as e:
            self.error(f"API request failed: {e}")
            return {
                "success": False,
                "url": None,
                "original_url": None,
                "filename": None,
                "local_path": None,
                "error": str(e)
            }
        except (KeyError, IndexError) as e:
            self.error(f"Invalid API response: {e}")
            return {
                "success": False,
                "url": None,
                "original_url": None,
                "filename": None,
                "local_path": None,
                "error": f"Invalid API response: {e}"
            }

    def _download_and_save(self, image_url: str, filename: Optional[str] = None) -> dict:
        """
        下载图片并保存到本地静态目录

        Args:
            image_url: 图片 URL
            filename: 自定义文件名（可选）

        Returns:
            dict: 包含 success, url, filename, local_path
        """
        try:
            # 下载图片
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            # 从 Content-Type 推断扩展名
            content_type = response.headers.get("Content-Type", "image/png")
            ext = self._get_extension_from_content_type(content_type)

            # 生成唯一文件名
            if not filename:
                unique_id = uuid.uuid4().hex[:8]
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"gen_{timestamp}_{unique_id}{ext}"

            # 保存文件
            file_path = IMAGES_DIR / filename
            with open(file_path, "wb") as f:
                f.write(response.content)

            self.info(f"Image saved: {file_path}")

            # 返回外部可访问的 URL (前端会通过 BACKEND_DIRECTORY 添加 /api 前缀)
            static_url = f"/static/images/{filename}"

            return {
                "success": True,
                "url": static_url,
                "filename": filename,
                "local_path": str(file_path)
            }

        except requests.RequestException as e:
            self.error(f"Failed to download image: {e}")
            return {
                "success": False,
                "url": None,
                "filename": None,
                "local_path": None,
                "error": str(e)
            }
        except IOError as e:
            self.error(f"Failed to save image: {e}")
            return {
                "success": False,
                "url": None,
                "filename": None,
                "local_path": None,
                "error": str(e)
            }

    def _get_extension_from_content_type(self, content_type: str) -> str:
        """根据 Content-Type 获取文件扩展名"""
        type_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/svg+xml": ".svg",
        }
        # 处理可能带有 charset 的 content-type
        main_type = content_type.split(";")[0].strip().lower()
        return type_map.get(main_type, ".png")

    def save_from_url(self, image_url: str, filename: Optional[str] = None) -> dict:
        """
        从外部 URL 下载图片并保存到本地
        便于其他模块调用

        Args:
            image_url: 图片 URL
            filename: 自定义文件名（可选）

        Returns:
            dict: 包含 success, url, filename, local_path
        """
        return self._download_and_save(image_url, filename)


if __name__ == "__main__":
    t = Text2ImageAPI()
    result = t.generate_text2image("A beautiful sunset over mountains")
    print(f"Result: {result}")

