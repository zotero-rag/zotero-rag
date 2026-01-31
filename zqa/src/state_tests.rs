
    #[test]
    fn test_oobe_skip_setup() {
        // Setup config dir
        let config_dir = get_config_dir().unwrap();
        fs::create_dir_all(&config_dir).unwrap();
        let config_path = config_dir.join("config.toml");
        if config_path.exists() {
            fs::remove_file(&config_path).unwrap();
        }

        // Simulate input: No to setup
        let input_str = "n\n";
        let mut input = Cursor::new(input_str);
        let mut output = Vec::new();

        let result = oobe(&mut input, &mut output);
        assert!(result.is_ok());

        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("Would you like to set up your configuration?"));
        // Config file should NOT be created if it didn't exist and we said no
        assert!(!config_path.exists());
    }

    #[test]
    fn test_oobe_alternative_providers() {
        // Setup config dir
        let config_dir = get_config_dir().unwrap();
        fs::create_dir_all(&config_dir).unwrap();
        let config_path = config_dir.join("config.toml");
        if config_path.exists() {
            fs::remove_file(&config_path).unwrap();
        }

        // Simulate input:
        // 1. Yes to setup
        // 2. Gemini ('g') for model
        // 3. API key
        // 4. Cohere ('c') for embedding
        // 5. API key
        // 6. Cohere ('c') for reranker
        // 7. API key
        // 8. Max requests (default)
        let input_str = "y\ng\ngemini-key\nc\ncohere-key\nc\ncohere-rerank-key\n\n";
        let mut input = Cursor::new(input_str);
        let mut output = Vec::new();

        let result = oobe(&mut input, &mut output);
        assert!(result.is_ok());

        let config = Config::from_file(&config_path).unwrap();
        assert_eq!(config.model_provider, "gemini");
        assert_eq!(config.embedding_provider, "cohere");
        assert_eq!(config.reranker_provider, "cohere");
        assert_eq!(config.gemini.unwrap().api_key, Some("gemini-key".to_string()));
        assert_eq!(config.cohere.unwrap().api_key, Some("cohere-rerank-key".to_string()));
    }
