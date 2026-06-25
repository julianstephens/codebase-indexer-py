class IndexerError(Exception):
    """Base class for all indexer errors."""

    pass


class QualifiedNameError(IndexerError):
    """Raised when a qualified name is invalid."""

    pass


class InvalidComputeArgumentsError(QualifiedNameError):
    """Raised when the arguments to compute() are invalid."""

    def __init__(
        self,
        file_path: str | None = None,
        name: str | None = None,
        parent: str | None = None,
    ):
        if file_path:
            self.file_path = file_path
            message = f"Invalid file path for qualified name: {file_path}"
        elif name:
            self.name = name
            message = f"Invalid symbol name for qualified name: {name}"
        elif parent:
            self.parent = parent
            message = f"Invalid parent name for qualified name: {parent}"
        else:
            message = "Invalid arguments for qualified name computation."
        super().__init__(message)


class FileExtensionNotSupportedError(IndexerError):
    """Raised when the extension of a file is not supported."""

    def __init__(self, extension: str):
        self.extension = extension
        message = f"File extension not supported: {extension}"
        super().__init__(message)


class StoreError(IndexerError):
    """Raised when there is an error with the store."""

    pass


class FileNotFoundError(StoreError):
    """Raised when a file is not found in the store."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        message = f"File not found in store: {file_path}"
        super().__init__(message)


class InvalidNodeRecordError(StoreError):
    """Raised when a node record is invalid."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = "Invalid node record"
        super().__init__(message)


class StoreOperationError(StoreError):
    """Raised when there is an error during a store operation."""

    def __init__(self, op: str | None = None, message: str | None = None):
        if message is None:
            message = "Store operation failed"
        if op:
            message = f"{message}: {op}"
        super().__init__(message)


class ArtifactError(StoreError):
    """Raised when there is an error with the artifact."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = "Artifact error"
        super().__init__(message)


class StoreFileNotFoundError(ArtifactError):
    """Raised when the database file is not found."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        message = f"Database file not found: {file_path}"
        super().__init__(message)


class InvalidStoreFileError(ArtifactError):
    """Raised when the database file is invalid."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        message = f"Invalid database file: {file_path}"
        super().__init__(message)


class ArtifactNotFoundError(ArtifactError):
    """Raised when the artifact file is not found."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        message = f"Artifact file not found: {file_path}"
        super().__init__(message)


class MetadataNotFoundError(ArtifactError):
    """Raised when the metadata file is not found."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        message = f"Metadata file not found: {file_path}"
        super().__init__(message)


class InvalidArtifactError(ArtifactError):
    """Raised when the artifact file is invalid."""

    def __init__(self, file_path: str, message: str | None = None):
        self.file_path = file_path
        if message is None:
            message = f"Invalid artifact file: {file_path}"
        super().__init__(message)


class ContextError(IndexerError):
    """Raised when there is an error with the context."""

    pass


class DatabaseNotFoundError(ContextError):
    """Raised when the database file is not found."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        message = f"Database file not found: {db_path}"
        super().__init__(message)


class InvalidContextError(ContextError):
    """Raised when the context is invalid."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = "Invalid context"
        super().__init__(message)


class QueryError(IndexerError):
    """Raised when there is an error with a query."""

    pass


class SearchQueryError(QueryError):
    """Raised when there is an error with a search query."""

    def __init__(self, query: str, message: str | None = None):
        self.query = query
        if message is None:
            message = f"Search query failed: {query}"
        super().__init__(message)


class EvaluationError(IndexerError):
    """Raised when there is an error with evaluation."""

    pass


class EvaluationSerializationError(EvaluationError):
    """Raised when an evaluation record cannot be serialized or parsed."""

    def __init__(self, message: str | None = None):
        if message is None:
            message = "Evaluation serialization error"
        super().__init__(message)


class UnsupportedTrajectoryEventError(EvaluationError):
    """Raised when an unsupported trajectory event is encountered."""

    def __init__(self, event_type: str):
        self.event_type = event_type
        message = f"Unsupported trajectory event type: {event_type}"
        super().__init__(message)
