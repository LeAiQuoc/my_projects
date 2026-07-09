from argostranslate import package
package.update_package_index()

available_packages = package.get_available_packages()
for pkg in available_packages:
    print(f"From: {pkg.from_code} To: {pkg.to_code}")


vi_en_package = next((pkg for pkg in available_packages if pkg.from_code == "vi" and pkg.to_code == "en"), None)
if vi_en_package:
    package.install_from_path(vi_en_package.download())
else:
    print("Vietnamese-to-English model not found in the package index.")


installed_languages = get_installed_languages()
for lang in installed_languages:
    print(f"Code: {lang.code}, Name: {lang.name}")



installed_languages = get_installed_languages()
try:
    vi_lang = next(lang for lang in installed_languages if lang.code == "vi")
    en_lang = next(lang for lang in installed_languages if lang.code == "en")
    translator = vi_lang.get_translation(en_lang)
except StopIteration:
    print("Vietnamese or English language model not installed. Please install them.")
    exit(1)


import requests
response = requests.post("https://libretranslate.com/translate", json={"q": "Xin chào", "source": "vi", "target": "en"})
print(response.json()["translatedText"])  # "Hello"


from argostranslate import package, translate

# Install Vietnamese model
package.update_package_index()
available_packages = package.get_available_packages()
vi_en_package = next((pkg for pkg in available_packages if pkg.from_code == "vi" and pkg.to_code == "en"), None)
if vi_en_package:
    package.install_from_path(vi_en_package.download())

# Your existing code
installed_languages = translate.get_installed_languages()
vi_lang = next(lang for lang in installed_languages if lang.code == "vi")
en_lang = next(lang for lang in installed_languages if lang.code == "en")
translator = vi_lang.get_translation(en_lang)
print(translator.translate("Xin chào"))  # Should print "Hello"