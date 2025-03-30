#include "config.hpp"
#include "../exceptions/exceptions.hpp"
#include "../io/FileReader.hpp"
#include "../json/JsonHelper.hpp"
#include "tag_invokes/tag_invokes.hpp"

namespace json = boost::json;

namespace config {

Config parse_json(const std::string &json_content) {
    try {
        json::value jv = JsonHelper::parse_json(json_content);
        Config config = json::value_to<Config>(jv);

        return config;
    } catch (const std::exception &e) {
        std::throw_with_nested(
            ConfigurationException("Error while parsing JSON configuration"));
    }
}

Config load(const std::string &filename) {
    FileReader reader(filename);
    std::string json_content;

    try {
        reader.read_all(json_content);
    } catch (const std::exception &e) {
        std::throw_with_nested(ConfigurationException("Could not read config file"));
    }

    return parse_json(json_content);
}
} // namespace config